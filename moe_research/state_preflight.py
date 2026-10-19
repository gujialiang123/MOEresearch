"""P0 correctness gate for the original BF16 Qwen state-source study.

All numerical thresholds are fixed before the first forward. Short generations
are engineering checks only. No calibration is fitted and no hypothesis is tested
here. A failed gate writes FAILED_IMPLEMENTATION and raises PreflightFailure.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import marshal
import platform
from pathlib import Path
import time
import traceback

import torch

from .gain_calibration import GainCalibrator
from .k_policy import KPolicy, _phase_from_cache, attach_policy, detach_policy
from .state_adapter import ADAPTER_NAME, route_and_aggregate, shadow_branches, validate_block_layout
from .state_calibration import FrozenStateCalibration
from .state_driver import SEED, StudyDriver, digest, write_json

TOLERANCES = {
    "logits_max_abs": 1e-4,
    "local_max_abs": 1e-4,
    "router_weights_max_abs": 1e-7,
    "argmax_agreement_min": 1.0,
    "fixed_prefix_kl_max": 1e-8,
}
PREFLIGHT_CAP = 8


class PreflightFailure(RuntimeError):
    pass


def tensor_errors(reference, actual):
    if reference.shape != actual.shape:
        raise PreflightFailure(f"Tensor shape mismatch {reference.shape} != {actual.shape}")
    left, right = reference.detach().float(), actual.detach().float()
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise PreflightFailure("Non-finite values in a correctness comparison")
    error = right - left
    return {
        "max_abs": float(error.abs().max()) if error.numel() else 0.0,
        "relative_l2": float(error.double().norm() / left.double().norm().clamp_min(1e-30)),
        "max_abs_relative_to_reference_max": float(error.abs().max() / left.abs().max().clamp_min(1e-30)) if error.numel() else 0.0,
        "elements": left.numel(),
    }


def compare_logits(reference, actual):
    if len(reference) != len(actual) or not reference:
        raise PreflightFailure("Logit sequence lengths differ or are empty")
    left, right = torch.stack(reference), torch.stack(actual)
    result = tensor_errors(left, right)
    lp, lq = left.double().log_softmax(-1), right.double().log_softmax(-1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    result.update({
        "fixed_prefix_kl_mean": float(kl.mean()),
        "fixed_prefix_kl_max": float(kl.max()),
        "argmax_agreement": float((left.argmax(-1) == right.argmax(-1)).float().mean()),
        "positions": len(reference),
    })
    return result


def assert_logit_gate(metrics):
    if (metrics["max_abs"] > TOLERANCES["logits_max_abs"] or
            metrics["fixed_prefix_kl_max"] > TOLERANCES["fixed_prefix_kl_max"] or
            metrics["argmax_agreement"] < TOLERANCES["argmax_agreement_min"]):
        raise PreflightFailure(f"Fixed logit equivalence threshold failed: {metrics}")


class _Capture:
    """Temporary CPU snapshots from ordinary hooks, independent of observers."""
    def __init__(self, driver):
        self.driver = driver
        self.records = {}
        self.phases = []
        self.residuals = {}
        self.handles = []

    def __enter__(self):
        def top_hook(module, args, kwargs):
            self.phases.append(_phase_from_cache(kwargs))
        self.handles.append(self.driver.model.register_forward_pre_hook(top_hook, with_kwargs=True))
        # Ordinary generation deliberately has no production residual observer.
        # These independent engineering-only hooks cover its actual residuals
        # without installing or sharing the diagnostic driver's hook state.
        for layer, decoder in enumerate(self.driver.model.model.layers):
            def residual_hook(module, args, layer=layer):
                self.residuals[layer] = args[0].detach().cpu().clone()
            self.handles.append(decoder.post_attention_layernorm.register_forward_pre_hook(residual_hook))
        for layer, block in enumerate(self.driver.blocks):
            def local_hook(module, args, output, layer=layer):
                if not isinstance(output, tuple) or len(output) != 2:
                    raise PreflightFailure("Unsupported MoE forward return format")
                h, y, router = args[0], output[0], output[1]
                p, selected = router.float().softmax(-1).topk(module.top_k, -1)
                weights = (p / p.sum(-1, keepdim=True)).to(h.dtype) if module.norm_topk_prob else p.to(h.dtype)
                key = (len(self.phases) - 1, layer)
                self.records[key] = {
                    "h": h.detach().cpu().clone(), "output": y.detach().cpu().clone(),
                    "router_logits": router.detach().cpu().clone(),
                    "native_selected": selected.detach().cpu().clone(),
                    "native_weights": weights.detach().cpu().clone(),
                    "residual": self.residuals[layer],
                }
            self.handles.append(block.register_forward_hook(local_hook))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def validate(self):
        expected = {(step, layer) for step in range(len(self.phases)) for layer in range(len(self.driver.blocks))}
        if set(self.records) != expected or self.phases != ["prefill"] + ["decode"] * (len(self.phases) - 1):
            raise PreflightFailure("Local capture did not cover every layer and cached decode forward")


def compare_captures(reference, actual, *, prefill_only=False):
    reference.validate(); actual.validate()
    keys = [key for key in reference.records if not prefill_only or key[0] == 0]
    if not prefill_only and reference.phases != actual.phases:
        raise PreflightFailure("Prefill/decode phase sequence differs")
    if any(key not in actual.records for key in keys):
        raise PreflightFailure("Missing local layer/step comparison")
    rows = []
    for step, layer in sorted(keys):
        left, right = reference.records[(step, layer)], actual.records[(step, layer)]
        row = {"forward_index": step, "phase": reference.phases[step], "layer": layer}
        for field in ("h", "output", "router_logits", "native_weights", "residual"):
            error = tensor_errors(left[field], right[field])
            row[field] = error
            tol = TOLERANCES["router_weights_max_abs"] if field == "native_weights" else TOLERANCES["local_max_abs"]
            if error["max_abs"] > tol:
                raise PreflightFailure(f"Local equivalence failed layer={layer} step={step} field={field}: {error}")
        row["router_choice_agreement"] = float((left["native_selected"] == right["native_selected"]).float().mean())
        if row["router_choice_agreement"] != 1.0:
            raise PreflightFailure("Native router top-k choice disagreement")
        rows.append(row)
    return {"layer_forward_comparisons": len(rows),
            "max_abs": max((value[field]["max_abs"] for value in rows for field in
                            ("h", "output", "router_logits", "native_weights", "residual")), default=0.0),
            "rows": rows}


def synthetic_calibrations(driver):
    """Non-unit gains and nonzero per-dimension LDA bias; never scientific fits."""
    dim = driver.model.config.hidden_size
    scalar = {str(layer): {"scale": 1.09 + layer * 0.0001} for layer in range(len(driver.blocks))}
    red_mu = torch.linspace(-0.003, 0.003, dim).tolist()
    native_mu = torch.linspace(0.002, -0.002, dim).tolist()
    red_std = torch.linspace(0.8, 1.2, dim).tolist()
    native_std = torch.linspace(0.9, 1.1, dim).tolist()
    lda = {str(layer): {"mean_reduced": red_mu, "mean_native": native_mu,
                       "std_reduced": red_std, "std_native": native_std}
           for layer in range(len(driver.blocks))}
    meta = {"purpose": "P0 synthetic nontrivial parameters; not fitted or used in scientific experiments"}
    return {"M_N": FrozenStateCalibration("scalar", scalar, metadata=meta),
            "L_N": FrozenStateCalibration("lda", lda, metadata=meta)}


def _interface(driver):
    import transformers
    blocks = driver.blocks
    first = blocks[0]
    validate_block_layout(first)
    expert = first.experts[0]
    try:
        import kernels
        kernel_version = getattr(kernels, "__version__", "unknown")
    except ImportError:
        kernel_version = None
    checkpoint_path = Path(driver.model.config._name_or_path)
    provenance_path = checkpoint_path / "study_download_provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    from . import k_policy, state_adapter, state_calibration, state_driver
    modules = {"state_preflight": Path(__file__), "k_policy": Path(k_policy.__file__),
               "state_adapter": Path(state_adapter.__file__),
               "state_calibration": Path(state_calibration.__file__),
               "state_driver": Path(state_driver.__file__)}
    functions = {
        "StudyDriver.__init__": StudyDriver.__init__, "StudyDriver.generate": StudyDriver.generate,
        "StudyDriver.replay": StudyDriver.replay, "StudyDriver.policy": StudyDriver.policy,
        "KPolicy.__post_init__": KPolicy.__post_init__,
        "_make_moe_forward": k_policy._make_moe_forward, "attach_policy": attach_policy,
        "detach_policy": detach_policy, "execute_branches": state_adapter.execute_branches,
        "route_and_aggregate": route_and_aggregate, "shadow_branches": shadow_branches,
        "FrozenStateCalibration.apply": FrozenStateCalibration.apply,
    }
    return {
        "model_class": type(driver.model).__name__, "block_class": type(first).__name__,
        "gate_class": type(first.gate).__name__, "experts_class": type(first.experts).__name__,
        "expert_class": type(expert).__name__,
        "expert_linears": {name: {"class": type(getattr(expert, name)).__name__,
                                   "weight_shape": list(getattr(expert, name).weight.shape)}
                           for name in ("gate_proj", "up_proj", "down_proj")},
        "gate_weight_shape": list(first.gate.weight.shape), "block_count": len(blocks),
        "num_experts": first.num_experts, "native_k": first.top_k,
        "norm_topk_prob": first.norm_topk_prob, "adapter": ADAPTER_NAME,
        "dtype": str(next(driver.model.parameters()).dtype),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "kernels": kernel_version, "cuda": torch.version.cuda,
        "python": platform.python_version(), "device": str(driver.device),
        "gpu": torch.cuda.get_device_name(driver.device),
        "attention_backend": driver.model.config._attn_implementation,
        "device_map": {k: str(v) for k, v in getattr(driver.model, "hf_device_map", {}).items()},
        "checkpoint_path": str(driver.model.config._name_or_path),
        "checkpoint_revision": provenance.get("revision"),
        "source_fingerprints": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for name, path in modules.items()},
        "loaded_code_fingerprints": {name: hashlib.sha256(marshal.dumps(inspect.unwrap(fn).__code__)).hexdigest()
                                     for name, fn in functions.items()},
        "eos_ids": sorted(driver.eos), "seed": SEED,
        "phase_detection": "cache-state; empty cache prefill; nonempty cache decode",
        "block_return_format": "(native_dtype_output, router_logits)",
        "gate_return_format": "[tokens, experts] Tensor",
        "expert_selection_rule": "Native top-8 ranking for every K4 branch, independent of reference-k; larger reference masses use a separate router-only top-k and never select FFNs",
        "floating_point_order": "FP32 router softmax -> native selection-pool top-k independent of reference-k -> separate reference probability mass when reference exceeds selection pool -> FP32 weight normalization -> cast final weights to model dtype -> expert multiply and ascending-expert index_add in model dtype -> FP32 postaggregate scalar/LDA -> cast to model dtype",
    }


def _shadow_observer(arm, calibrations, rows):
    def observe(block, h, actual, phase, layer, step, route_info):
        if phase != "decode":
            expected = route_and_aggregate(block, h, block.top_k, block.top_k)[0]
        else:
            branches = shadow_branches(block, h)
            if arm == "N":
                expected = branches["y8"]
            elif arm == "F":
                expected = branches["y4_full"]
            elif arm == "R8":
                expected = branches["u4"]
            elif arm == "R16":
                expected = route_and_aggregate(block, h, 4, 16)[0]
            else:
                branch = branches["u4"] if arm == "M_N" else branches["y4_full"]
                expected = calibrations[arm].apply(layer, branch, phase)
        error = tensor_errors(expected, actual)
        row = {"arm": arm, "phase": phase, "layer": layer, "decode_step": step, **error}
        if error["max_abs"] > TOLERANCES["local_max_abs"]:
            raise PreflightFailure(f"Shadow-real aggregation mismatch {row}")
        if phase == "decode" and arm != "N":
            kc, kr = 4, {"F": 4, "R8": 8, "R16": 16, "M_N": 8, "L_N": 4}[arm]
            logits = block.gate(h.reshape(-1, h.shape[-1]))
            probabilities = logits.float().softmax(-1)
            # Reference-k must not change the tie-breaking pool used to select
            # the executed experts. Match the fixed native top-8 rule explicitly.
            selection_pool_k = max(block.top_k, kc)
            p, selected = probabilities.topk(selection_pool_k, -1)
            ref_prob = p[:, :kr] if kr <= selection_pool_k else probabilities.topk(kr, -1).values
            mass = ref_prob.sum(-1, keepdim=True)
            weights = (p[:, :kc] / mass).to(h.dtype)
            row["router_weights"] = tensor_errors(weights, route_info["weights"])
            row["reference_mass"] = tensor_errors(mass, route_info["reference_mass"])
            row["selected_match"] = bool(torch.equal(selected[:, :kc], route_info["selected"]))
            if (row["router_weights"]["max_abs"] > TOLERANCES["router_weights_max_abs"] or
                    row["reference_mass"]["max_abs"] > TOLERANCES["router_weights_max_abs"] or
                    not row["selected_match"]):
                raise PreflightFailure(f"Actual routing weight mismatch {row}")
        rows.append(row)
    return observe


def _samples():
    from datasets import load_dataset
    dataset = load_dataset("gsm8k", "main", split="train[128:132]")
    return [{"sample_id": f"gsm8k:train:{128 + i}", "question_hash": digest(row["question"]),
             "question": row["question"], "answer": row["answer"]}
            for i, row in enumerate(dataset)]


@torch.inference_mode()
def run_preflight(driver: StudyDriver, out_dir, *, samples=None):
    """Run all P0 gates and return the persisted report, or raise on failure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "preflight.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get("status") == "PASSED":
            raise RuntimeError("Preflight raw results already exist; use a new output directory")
        raise RuntimeError("Previous incomplete/failed preflight preserved; choose a new output directory")
    report = {"schema_version": "state_preflight.v1", "status": "RUNNING",
              "tolerances": dict(TOLERANCES), "engineering_cap": PREFLIGHT_CAP,
              "scientific_results": False, "started_unix": time.time(), "cases": []}
    write_json(report_path, report)
    def checkpoint(name, sample_id, metrics):
        record = {"name": name, "sample_id": sample_id, **metrics}
        report["cases"].append(record)
        write_json(report_path, report)
        print(f"P0 {sample_id} {name}: PASS", flush=True)
    try:
        report["interface"] = _interface(driver)
        if (len(driver.blocks) != 48 or next(driver.model.parameters()).dtype != torch.bfloat16 or
                any(not block.norm_topk_prob for block in driver.blocks)):
            raise PreflightFailure("Real preflight requires original 48-layer BF16 native-normalized Qwen")
        samples = _samples() if samples is None else list(samples)
        if len(samples) != 4:
            raise PreflightFailure("Correctness smoke must contain exactly four prompts")
        report["samples"] = [{k: sample[k] for k in ("sample_id", "question_hash", "question")} for sample in samples]
        calibrations = synthetic_calibrations(driver)
        write_json(out_dir / "synthetic_calibrations.json", {arm: cal.to_dict() for arm, cal in calibrations.items()})
        for sample in samples:
            sid = sample["sample_id"]
            with _Capture(driver) as native_capture:
                native, native_logits = driver.generate(sample, "N", {}, cap=PREFLIGHT_CAP, capture_logits=True)
            native_capture.validate()
            write_json(out_dir / f"{digest(sid)}.native_trace.json", native)
            for name, forced in (("native_repeat_noise", False), ("forced_native_executor", True),
                                 ("detach_restores_native", False)):
                with _Capture(driver) as capture:
                    row, logits = driver.generate(sample, "N", {}, cap=PREFLIGHT_CAP,
                                                  capture_logits=True, force_native=forced)
                metric = compare_logits(native_logits, logits)
                assert_logit_gate(metric)
                if row["gen_token_ids"] != native["gen_token_ids"]:
                    raise PreflightFailure(f"{name}: short greedy output differs")
                local = compare_captures(native_capture, capture)
                checkpoint(name, sid, {"logits": metric, "local": local,
                                       "short_greedy_match": True})
                del capture, logits
            # Verify the manual protocol against the checkpoint's HF generator.
            _, ids = driver.prompt(sample)
            torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            hf = driver.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=PREFLIGHT_CAP, do_sample=False,
                eos_token_id=sorted(driver.eos), pad_token_id=driver.tokenizer.pad_token_id)
            hf_tokens = hf[0, ids.shape[1]:].tolist()
            if hf_tokens != native["gen_token_ids"]:
                raise PreflightFailure("HF generate greedy tokens differ from manual driver")
            checkpoint("hf_generate_matches_manual", sid, {"short_greedy_match": True,
                                                            "gen_token_ids": hf_tokens})
            del hf
            for arm in ("N", "F", "R8", "R16", "M_N", "L_N"):
                if arm == "N":
                    generated, gen_logits, gen_capture = native, native_logits, native_capture
                else:
                    with _Capture(driver) as gen_capture:
                        generated, gen_logits = driver.generate(sample, arm, calibrations,
                            cap=PREFLIGHT_CAP, capture_logits=True)
                    write_json(out_dir / f"{digest([sid, arm])}.trace.json", generated)
                pref = compare_captures(native_capture, gen_capture, prefill_only=True)
                shadow_rows = []
                observer = _shadow_observer(arm, calibrations, shadow_rows)
                with _Capture(driver) as replay_capture:
                    replay, replay_logits = driver.replay(sample, generated["gen_token_ids"], arm,
                        calibrations, observer=observer, capture_logits=True)
                metric = compare_logits(gen_logits, replay_logits)
                assert_logit_gate(metric)
                if replay["argmaxes"] != generated["gen_token_ids"]:
                    raise PreflightFailure("Cached replay does not reproduce its arm's unobserved greedy trace")
                local = compare_captures(gen_capture, replay_capture)
                expected_rows = len(driver.blocks) * generated["generated_tokens"]
                if len(shadow_rows) != expected_rows:
                    raise PreflightFailure("Shadow observer missed a layer/forward")
                checkpoint(f"{arm}_shadow_real_and_observer_replay", sid, {
                    "logits": metric, "local": local, "prefill_native": pref,
                    "shadow_real_rows": shadow_rows,
                    "replay_path": "A" if arm == "N" else ("C" if arm == "L_N" else "arm_own_trace"),
                    "replay_argmaxes_match": True, "decode_forwards": replay["decode_forwards"],
                    "synthetic_calibration": arm in calibrations,
                })
                del replay_capture, replay_logits, shadow_rows
                if arm != "N":
                    del gen_capture, gen_logits
            # Real non-unit parameters and a real RNG-backed shuffled provider
            # exercise native identity and both selector dimensions explicitly.
            for window in ("native_identity", "unselected_layer", "unselected_step"):
                cal = GainCalibrator([], low_k=4)
                cal.pools = {(layer, 0): [1.8, 2.2] for layer in range(len(driver.blocks))}
                policy = KPolicy(
                    prefill_k=8, decode_k=8 if window == "native_identity" else 4,
                    weight_mode="shuffled_gain", gain_provider=cal.make_shuffled_provider(SEED),
                    calib_scalars={f"{layer},4": 2.7 for layer in range(len(driver.blocks))},
                    output_transform=calibrations["L_N"].output_transform,
                    layer_selector=(lambda layer: False) if window == "unselected_layer" else None,
                    decode_step_selector=(lambda step: False) if window == "unselected_step" else None,
                )
                context = attach_policy(driver.model, policy)
                try:
                    with _Capture(driver) as capture:
                        generated, logits = driver.generate(sample, "N", {}, cap=2, capture_logits=True)
                finally:
                    detach_policy(driver.model, context)
                metric = compare_logits(native_logits[:len(logits)], logits)
                assert_logit_gate(metric)
                if generated["gen_token_ids"] != native["gen_token_ids"][:2]:
                    raise PreflightFailure(f"Nontrivial calibration leaked into {window}")
                # Compare the two captured calls against the matching prefix.
                old_records, old_phases = native_capture.records, native_capture.phases
                native_capture.records = {key: value for key, value in old_records.items() if key[0] < len(logits)}
                native_capture.phases = old_phases[:len(logits)]
                try:
                    local = compare_captures(native_capture, capture)
                finally:
                    native_capture.records, native_capture.phases = old_records, old_phases
                checkpoint(window, sid, {"logits": metric, "local": local,
                                         "nonzero_bias": True, "nonunit_scalar": 2.7,
                                         "real_shuffled_provider": True})
            del native_capture, native_logits
            torch.cuda.empty_cache()
        report["status"] = "PASSED"
        report["completed_unix"] = time.time()
        report["case_count"] = len(report["cases"])
        report["calibration_fit_used"] = False
        report["cache_isolation"] = "Each generation/replay initializes an independent cache; every arm attaches/detaches its policy."
        write_json(report_path, report)
        print(f"P0 PASSED: {report_path}", flush=True)
        return report
    except Exception as exc:
        report["status"] = "FAILED_IMPLEMENTATION"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        report["completed_unix"] = time.time()
        write_json(report_path, report)
        print(f"P0 FAILED_IMPLEMENTATION: {exc}", flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/scratch/jgu7/models/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"P0 loading BF16 checkpoint {args.model} on {args.device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="eager", device_map={"": args.device}).eval()
    print("P0 model loaded", flush=True)
    driver = StudyDriver(model, tokenizer)
    try:
        run_preflight(driver, args.out)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
