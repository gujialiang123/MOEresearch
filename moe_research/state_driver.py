"""Single-request greedy generation and cached replay for the state study.

No observer is used during measured free generation. Replay initializes its own
cache, returns its actual driver branch, and consumes exactly trace[:-1].
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import time
import torch
from . import answer_parsing as AP
from .k_policy import KPolicy, attach_policy, detach_policy, _find_moe_blocks
from .state_adapter import shadow_branches

SEED = 20261019
SUFFIX = "\nPlease reason step by step, and put your final answer after '#### '."
PARSER_VERSION = hashlib.sha256(Path(AP.__file__).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.next')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def append_jsonl(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def read_rows(path, expected=None):
    path = Path(path)
    if not path.exists():
        return {}
    rows = {}
    with path.open() as f:
        for line, text in enumerate(f, 1):
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'Truncated/invalid JSONL {path}:{line}; preserve and repair explicitly') from exc
            key = row['sample_id']
            if key in rows:
                raise RuntimeError(f'Duplicate sample {key} in {path}')
            for k, v in (expected or {}).items():
                if row.get(k) != v:
                    raise RuntimeError(f'Resume identity mismatch: {path}:{line} field {k}')
            rows[key] = row
    return rows


def arm_policy(arm, calibrations, observer=None, force_native=False):
    if arm == 'N':
        return KPolicy(prefill_k=8, decode_k=8, observer=observer,
                       force_native_execution=force_native)
    reference = {'F': 4, 'R8': 8, 'R16': 16, 'M_N': 8, 'L_N': 4,
                 'M_NT': 4, 'M_ST': 4}[arm]
    calibration = calibrations.get(arm)
    if arm in ('M_N','L_N','M_NT','M_ST') and calibration is None:
        raise ValueError(f'{arm} requires a frozen calibration; refusing an uncalibrated mislabeled arm')
    if arm in ('M_NT', 'M_ST') and calibration is not None:
        reference = 8 if calibration.family == 'scalar' else 4
    return KPolicy(prefill_k=8, decode_k=4, k_compute=4, k_reference=reference,
                   weight_mode='reference_k', observer=observer,
                   output_transform=calibration.output_transform if calibration else None)


def arm_hash(arm, calibrations):
    reference = {'N':8,'F':4,'R8':8,'R16':16,'M_N':8,'L_N':4,'M_NT':4,'M_ST':4}[arm]
    if arm in ('M_NT','M_ST') and arm in calibrations:
        reference = 8 if calibrations[arm].family == 'scalar' else 4
    return digest({'arm': arm, 'k_compute': 8 if arm == 'N' else 4,
                   'k_reference': reference,
                   'calibration_hash': calibrations[arm].hash if arm in calibrations else None,
                   'prefill': 'native', 'seed': SEED})


def eos_ids(model, tokenizer):
    result = set()
    for obj in (model.generation_config, tokenizer):
        value = getattr(obj, 'eos_token_id', None)
        if value is not None:
            result.update(value if isinstance(value, (list, tuple)) else [value])
    if not result:
        raise RuntimeError('Checkpoint has no resolved EOS IDs')
    return result


def validate_finite_logits(logits, *, arm, sample_id, forward_index, operation):
    """Reject NaN/Inf before token selection without modifying finite logits.

    This common check runs for every arm and every generation/replay forward.
    Its reduction/synchronization is included in driver wall time; existing
    prefill timing boundaries and token selection operations are unchanged.
    """
    if not bool(torch.isfinite(logits).all()):
        phase = "prefill" if forward_index == 0 else "decode"
        raise FloatingPointError(
            f"Non-finite logits: operation={operation}, arm={arm}, "
            f"sample_id={sample_id!r}, forward_index={forward_index}, phase={phase}"
        )


class StudyDriver:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.device = next(model.parameters()).device
        self.blocks, _ = _find_moe_blocks(model)
        if len(self.blocks) != 48 or model.config.num_experts_per_tok != 8:
            raise RuntimeError('This protocol requires the original 48-layer native-K8 Qwen checkpoint')
        self.eos = eos_ids(model, tokenizer)
        self.residuals = {}
        self._residual_handles = []
    def _attach_residual_hooks(self):
        if self._residual_handles:
            raise RuntimeError('Residual observer already attached')
        for layer, decoder in enumerate(self.model.model.layers):
            def hook(module, args, layer=layer):
                # Qwen adds attention to its residual BEFORE this layernorm.
                # Its input is therefore the actual later MoE residual-add input.
                self.residuals[layer] = args[0].detach().clone()
            self._residual_handles.append(decoder.post_attention_layernorm.register_forward_pre_hook(hook))

    def prompt(self, sample):
        text = self.tokenizer.apply_chat_template(
            [{'role':'user','content':sample['question'] + SUFFIX}],
            tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        if encoded['input_ids'].shape[1] + 4096 > self.model.config.max_position_embeddings:
            raise ValueError('Prompt exceeds frozen no-truncation input policy')
        return text, encoded['input_ids'].to(self.device)

    @contextmanager
    def policy(self, arm, calibrations, observer=None, force_native=False):
        # The uninstrumented native reference is literally unmodified.
        if arm == 'N' and observer is None and not force_native:
            yield None
            return
        if observer is not None:
            self._attach_residual_hooks()
        ctx = None
        try:
            ctx = attach_policy(self.model, arm_policy(arm, calibrations, observer, force_native))
            yield ctx
        finally:
            if ctx is not None:detach_policy(self.model, ctx)
            if observer is not None:
                self.close()
                self.residuals.clear()

    @torch.inference_mode()
    def generate(self, sample, arm, calibrations, cap=2048, capture_logits=False,
                 observer=None, force_native=False):
        if not isinstance(cap,int) or cap<1:raise ValueError('cap must be a positive integer')
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        text, ids = self.prompt(sample)
        tokens, logits = [], []
        sync = lambda: torch.cuda.synchronize(self.device)
        with self.policy(arm, calibrations, observer, force_native) as ctx:
            torch.cuda.reset_peak_memory_stats(self.device)
            sync(); start = time.perf_counter()
            out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                             use_cache=True, logits_to_keep=1, return_dict=True)
            cache = out.past_key_values
            sync(); prefill_end = time.perf_counter()
            validate_finite_logits(out.logits, arm=arm, sample_id=sample['sample_id'],
                                   forward_index=0, operation='generate')
            if capture_logits:
                logits.append(out.logits[0,-1].float().detach().cpu())
            tokens.append(int(out.logits[0,-1].argmax()))
            while len(tokens) < cap and tokens[-1] not in self.eos:
                out = self.model(input_ids=torch.tensor([[tokens[-1]]],device=self.device),
                                 past_key_values=cache, use_cache=True,
                                 logits_to_keep=1, return_dict=True)
                cache = out.past_key_values
                validate_finite_logits(out.logits, arm=arm, sample_id=sample['sample_id'],
                                       forward_index=len(tokens), operation='generate')
                if capture_logits:
                    logits.append(out.logits[0,-1].float().detach().cpu())
                tokens.append(int(out.logits[0,-1].argmax()))
            sync(); end = time.perf_counter()
            counters = ctx.stats() if ctx else None
            memory = {'peak_allocated_bytes':torch.cuda.max_memory_allocated(self.device),
                      'peak_reserved_bytes':torch.cuda.max_memory_reserved(self.device)}
        raw = self.tokenizer.decode(tokens, skip_special_tokens=True)
        strict, strict_status = AP.parse_strict(raw)
        tolerant, tolerant_status = AP.parse_tolerant(raw)
        gold = AP.parse_gold(sample['answer'])
        marker_char = raw.find('####')
        # Decode prefixes so token boundaries spanning the marker are handled.
        marker_token = None
        if marker_char >= 0:
            for i in range(1, len(tokens)+1):
                if '####' in self.tokenizer.decode(tokens[:i],skip_special_tokens=True):
                    marker_token = i-1; break
        grams = [tuple(tokens[i:i+4]) for i in range(max(0,len(tokens)-3))]
        row = {'sample_id':sample['sample_id'], 'question_hash':sample['question_hash'],
               'question':sample['question'], 'rendered_prompt':text,
               'prompt_token_ids':ids[0].tolist(), 'prompt_tokens':ids.shape[1],
               'arm':arm,'arm_hash':arm_hash(arm,calibrations),
               'calibration_hash':calibrations[arm].hash if arm in calibrations else None,
               'raw_output':raw,'gen_token_ids':tokens,'generated_tokens':len(tokens),
               'decode_forward_count':max(0,len(tokens)-1),
               'eos_position':len(tokens)-1 if tokens[-1] in self.eos else None,
               'stop_reason':'eos' if tokens[-1] in self.eos else 'max_new_tokens',
               'max_new_tokens':cap,'eos_ids':sorted(self.eos),
               'gold':gold,'pred_strict':strict,'pred_tolerant':tolerant,
               'strict_parse_status':strict_status,'tolerant_parse_status':tolerant_status,
               'strict_correct':strict is not None and strict==gold,
               'tolerant_correct':tolerant is not None and tolerant==gold,
               'parser_disagreement':strict!=tolerant,
               'parser_version':PARSER_VERSION,
               'has_answer_marker':marker_char>=0,'first_marker_position':marker_token,
               'post_marker_tokens':len(tokens)-marker_token-1 if marker_token is not None else None,
               'repeated_4gram_fraction':1-len(set(grams))/len(grams) if grams else 0.0,
               'generation_seconds':end-start,'prefill_seconds':prefill_end-start,
               'decode_seconds':end-prefill_end,'observer_enabled':observer is not None,
               'timing_debug_logits':capture_logits,'policy_counters':counters,
               'logits_finiteness_check':'all returned logits before every argmax; included common driver overhead; unchanged prefill timing boundary',
               'expert_token_assignments_simplified':48*(8*ids.shape[1]+(8 if arm=='N' else 4)*max(0,len(tokens)-1)),
               'budget_definition':'expert-token assignments; prefill native; T-1 decode forwards; excludes shadow overhead',
               **memory}
        del out, cache
        return row, logits

    @torch.inference_mode()
    def replay(self, sample, trace, arm, calibrations, observer=None, capture_logits=False,
               force_native=False):
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        _, ids = self.prompt(sample)
        if not trace:
            raise ValueError('Empty generation trace')
        if any(t in self.eos for t in trace[:-1]):
            raise ValueError('Replay trace contains tokens after an emitted EOS')
        logits, argmaxes = [], []
        with self.policy(arm,calibrations,observer,force_native):
            out = self.model(input_ids=ids,attention_mask=torch.ones_like(ids),
                             use_cache=True,logits_to_keep=1,return_dict=True)
            cache = out.past_key_values
            validate_finite_logits(out.logits, arm=arm, sample_id=sample['sample_id'],
                                   forward_index=0, operation='replay')
            argmaxes.append(int(out.logits[0,-1].argmax()))
            if capture_logits:logits.append(out.logits[0,-1].float().detach().cpu())
            for forward_index, token in enumerate(trace[:-1], 1):
                out = self.model(input_ids=torch.tensor([[token]],device=self.device),
                                 past_key_values=cache,use_cache=True,logits_to_keep=1,return_dict=True)
                cache=out.past_key_values
                validate_finite_logits(out.logits, arm=arm, sample_id=sample['sample_id'],
                                       forward_index=forward_index, operation='replay')
                argmaxes.append(int(out.logits[0,-1].argmax()))
                if capture_logits:logits.append(out.logits[0,-1].float().detach().cpu())
        del out,cache
        return {'argmaxes':argmaxes,'decode_forwards':len(trace)-1,
                'logits_finiteness_check':'all returned logits before every argmax; included common driver overhead',
                'mask_hash':digest([[i,t] for i,t in enumerate(trace[:-1],1)])},logits

    def close(self):
        for handle in self._residual_handles:handle.remove()
        self._residual_handles=[]
