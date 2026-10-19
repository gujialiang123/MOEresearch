"""CPU-only cached-sequence and resume checks; no real model or CUDA work."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import torch

from moe_research import state_driver as driver_module
from moe_research.state_driver import StudyDriver, append_jsonl, read_rows, digest


class ScriptedModel:
    def __init__(self, predictions, logit_faults=None):
        self.predictions = predictions
        self.logit_faults = logit_faults or {}
        self.calls = []
        self.caches = []

    def __call__(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is None:
            cache = {"index":0}
            self.caches.append(cache)
        else:
            cache = past_key_values
            cache["index"] += 1
        index = cache["index"]
        self.calls.append({"input":input_ids.clone(), "cache":cache,
                           "cached":past_key_values is not None, "index":index})
        logits = torch.zeros(1,1,100)
        logits[0,0,self.predictions[min(index,len(self.predictions)-1)]] = 1
        if index in self.logit_faults:
            logits[0,0,2] = self.logit_faults[index]
        return types.SimpleNamespace(logits=logits,past_key_values=cache)


class TextTokenizer:
    def decode(self, ids, **kwargs):
        return "".join({10:"Work ",11:"#### ",17:"42",99:"",98:""}.get(i,str(i)) for i in ids)


def cpu_driver(predictions, logit_faults=None):
    driver = StudyDriver.__new__(StudyDriver)
    driver.model = ScriptedModel(predictions, logit_faults)
    driver.tokenizer = TextTokenizer()
    driver.device = torch.device("cpu")
    driver.eos = {98,99}
    driver.residuals = {}
    driver._residual_handles = []
    driver.prompt = lambda sample:("fixture prompt",torch.tensor([[1,2]]))
    @contextmanager
    def policy(*args,**kwargs):
        yield None
    driver.policy = policy
    return driver


class StateDriverTests(unittest.TestCase):
    def setUp(self):
        self.cuda = mock.patch.multiple(torch.cuda, manual_seed_all=mock.DEFAULT,
            synchronize=mock.DEFAULT, reset_peak_memory_stats=mock.DEFAULT,
            max_memory_allocated=mock.DEFAULT, max_memory_reserved=mock.DEFAULT)
        values = self.cuda.start()
        values["max_memory_allocated"].return_value=1000
        values["max_memory_reserved"].return_value=2000
        self.addCleanup(self.cuda.stop)
        self.sample={"sample_id":0,"question_hash":"fixture","question":"fixture?","answer":"#### 42"}

    def test_generation_and_replay_use_first_token_prefill_and_exact_trace_prefix(self):
        driver=cpu_driver([10,11,17,99])
        generation,logits=driver.generate(self.sample,"N",{},cap=4,capture_logits=True)
        self.assertEqual(generation["gen_token_ids"],[10,11,17,99])
        self.assertEqual(generation["decode_forward_count"],3)
        self.assertEqual(generation["stop_reason"],"eos")
        self.assertEqual(generation["eos_position"],3)
        self.assertTrue(generation["strict_correct"])
        self.assertEqual([call["input"].tolist() for call in driver.model.calls],
                         [[[1,2]],[[10]],[[11]],[[17]]])
        driver.model.calls.clear()
        replay,replayed=driver.replay(self.sample,generation["gen_token_ids"],"N",{},capture_logits=True)
        self.assertEqual(replay["argmaxes"],generation["gen_token_ids"])
        self.assertEqual(replay["decode_forwards"],3)
        self.assertEqual(replay["mask_hash"],digest([[1,10],[2,11],[3,17]]))
        self.assertEqual([call["input"].tolist() for call in driver.model.calls],
                         [[[1,2]],[[10]],[[11]],[[17]]])
        self.assertIsNot(driver.model.caches[0],driver.model.caches[1])
        for left,right in zip(logits,replayed):torch.testing.assert_close(left,right)

    def test_eos_exactly_at_cap_and_cap_without_eos_have_distinct_reasons(self):
        exact,_=cpu_driver([10,17,98]).generate(self.sample,"N",{},cap=3)
        capped,_=cpu_driver([10,17,11]).generate(self.sample,"N",{},cap=3)
        self.assertEqual(exact["stop_reason"],"eos")
        self.assertEqual(exact["eos_position"],2)
        self.assertEqual(capped["stop_reason"],"max_new_tokens")
        self.assertIsNone(capped["eos_position"])
        self.assertEqual(exact["generated_tokens"],capped["generated_tokens"])

    def test_fixed_text_replay_does_not_stop_at_hypothetical_earlier_eos(self):
        driver=cpu_driver([99,99,99])
        result,_=driver.replay(self.sample,[10,17,11],"N",{})
        self.assertEqual(result["argmaxes"],[99,99,99])
        self.assertEqual(result["decode_forwards"],2)
        self.assertEqual([call["input"].tolist() for call in driver.model.calls],
                         [[[1,2]],[[10]],[[17]]])
        with self.assertRaises(ValueError):driver.replay(self.sample,[10,99,17],"N",{})

    def test_resume_rejects_duplicate_truncated_and_wrong_identity_rows(self):
        scratch=Path("/scratch/jgu7/tmp");scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path=Path(directory)/"raw.jsonl"
            record={"sample_id":1,"protocol_hash":"p","arm_hash":"a"}
            append_jsonl(path,record)
            self.assertEqual(read_rows(path,{"protocol_hash":"p","arm_hash":"a"}),{1:record})
            with self.assertRaisesRegex(RuntimeError,"identity mismatch"):
                read_rows(path,{"protocol_hash":"different"})
            append_jsonl(path,record)
            with self.assertRaisesRegex(RuntimeError,"Duplicate"):
                read_rows(path)
            path.write_text('{"sample_id":')
            with self.assertRaisesRegex(RuntimeError,"invalid JSONL"):
                read_rows(path)
            self.assertEqual(path.read_text(),'{"sample_id":')

    def test_calibrated_arms_require_a_frozen_calibration(self):
        for arm in ("M_N","L_N","M_NT","M_ST"):
            with self.subTest(arm=arm), self.assertRaises((ValueError,KeyError,RuntimeError)):
                driver_module.arm_policy(arm,{})

    def test_failed_policy_attachment_removes_diagnostic_hooks(self):
        driver=StudyDriver.__new__(StudyDriver)
        driver.model=object();driver._residual_handles=[];driver.residuals={}
        handle=mock.Mock()
        driver._attach_residual_hooks=lambda:driver._residual_handles.append(handle)
        with mock.patch.object(driver_module,"attach_policy",side_effect=RuntimeError("unsupported layout")):
            with self.assertRaisesRegex(RuntimeError,"unsupported layout"):
                with StudyDriver.policy(driver,"N",{},observer=object()):pass
        self.assertEqual(driver._residual_handles,[])
        handle.remove.assert_called_once()

    def test_nonfinite_logits_stop_generate_and_replay_with_forward_context(self):
        for operation in ("generate", "replay"):
            for index in (0,1):
                for bad in (float("nan"),float("inf"),float("-inf")):
                    with self.subTest(operation=operation,index=index,bad=bad):
                        driver=cpu_driver([10,11,17,99],{index:bad})
                        arm="R16" if operation=="generate" else "N"
                        phase="prefill" if index==0 else "decode"
                        with self.assertRaises(FloatingPointError) as caught:
                            if operation=="generate":
                                driver.generate(self.sample,arm,{},cap=4,capture_logits=True)
                            else:
                                driver.replay(self.sample,[10,11,17,99],arm,{},capture_logits=True)
                        message=str(caught.exception)
                        for context in (f"operation={operation}",f"arm={arm}","sample_id=0",
                                        f"forward_index={index}",f"phase={phase}"):
                            self.assertIn(context,message)
                        # The corrupt forward must not be followed by another
                        # model call or silently converted into a success row.
                        self.assertEqual(len(driver.model.calls),index+1)

    def test_finite_validation_preserves_logits_and_normal_token_protocol(self):
        for dtype in (torch.float32,torch.bfloat16):
            logits=torch.tensor([[[0.25,-3,5]]],dtype=dtype)
            original=logits.clone()
            driver_module.validate_finite_logits(logits,arm="N",sample_id="finite",
                                                  forward_index=0,operation="generate")
            self.assertTrue(torch.equal(logits,original))
            self.assertEqual(logits.dtype,dtype)
        driver=cpu_driver([10,11,17,99])
        row,logits=driver.generate(self.sample,"R16",{},cap=4,capture_logits=True)
        self.assertEqual(row["gen_token_ids"],[10,11,17,99])
        self.assertEqual(row["stop_reason"],"eos")
        self.assertEqual(row["decode_forward_count"],3)
        self.assertIn("common driver overhead",row["logits_finiteness_check"])
        replay,replayed=driver.replay(self.sample,row["gen_token_ids"],"R16",{},capture_logits=True)
        self.assertEqual(replay["argmaxes"],row["gen_token_ids"])
        self.assertIn("common driver overhead",replay["logits_finiteness_check"])
        for left,right in zip(logits,replayed):
            torch.testing.assert_close(left,right,rtol=0,atol=0)


if __name__=="__main__":unittest.main()
