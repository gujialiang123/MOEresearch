"""CPU-only analysis of archived state-source diagnostics.

This module does not fit calibration parameters, change a policy, or inspect
gold answers. It rebuilds nonlinear endpoints from per-prompt sufficient
statistics and uses the predeclared bounds solely as scientific decision rules.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .state_collection import (
    StateCollector, _bootstrap_intervals, _evaluate, _prepare, assert_matched_masks,
)


PATH_ALIASES = {
    "A": ("A", "A_native_native_text"),
    "B": ("B", "B_student_native_text"),
    "C": ("C", "C_student_student_text"),
    "D": ("D", "D_native_student_text"),
    "M_NT": ("M_NT", "S_NT", "S_NT_own", "M_NT_own"),
    "M_ST": ("M_ST", "S_ST", "S_ST_own", "M_ST_own"),
}


def _path_aliases(collector):
    available = sorted({key[1] for key in collector.records} |
                       {key[1] for key in collector.empty_records})
    resolved = {}
    for canonical, aliases in PATH_ALIASES.items():
        matches = [name for name in aliases if name in available]
        if len(matches) > 1:
            raise ValueError(f"ambiguous diagnostic paths for {canonical}: {matches}")
        if matches:
            resolved[canonical] = matches[0]
    return resolved, available


def _coverage(collector, resolved, group):
    output = {}
    for canonical, actual in resolved.items():
        rows = [(key, stats) for key, stats in collector.records.items()
                if key[1] == actual and key[3] == group]
        counts = {}
        masks = {}
        for (pid, _, layer, _), stats in rows:
            counts.setdefault(pid, {})[str(layer)] = stats.count
            masks.setdefault(pid, {})[str(layer)] = stats.mask_hash
        first_layer_counts = [values[min(values, key=int)] for values in counts.values()]
        empty_ids = sorted(pid for pid, marker_path in collector.empty_records if marker_path == actual)
        universe = ({key[0] for key in collector.records if key[1] == actual} | set(empty_ids))
        output[canonical] = {
            "path": actual, "status": "observed" if rows else "NO_OBSERVATIONS",
            "prompt_count": len(counts), "effective_position_count": sum(first_layer_counts),
            "layer_token_count": sum(stats.count for _, stats in rows),
            "counts_per_prompt_layer": counts, "mask_hashes_per_prompt_layer": masks,
            "zero_observation_prompt_count": len(empty_ids),
            "zero_observation_prompt_ids": empty_ids,
            "total_recorded_prompt_count": len(universe),
            "bootstrap_prompt_count": len(universe),
            "inactive_in_group_prompt_ids": sorted(universe - set(counts)),
        }
    return output


def _same_position_coverage(packs):
    reference = packs[0]
    for pack in packs[1:]:
        if pack["prompts"] != reference["prompts"] or pack["layers"] != reference["layers"]:
            return False
        for pid in reference["prompts"]:
            for layer in reference["layers"]:
                left_row = reference["rows"].get((pid, reference["path"], layer, reference["group"]))
                right_row = pack["rows"].get((pid, pack["path"], layer, pack["group"]))
                left = left_row.positions if left_row is not None else []
                right = right_row.positions if right_row is not None else []
                if [index for index, _ in left] != [index for index, _ in right]:
                    return False
    return True


def _pair_valid(collector, left, right, group, packs, *, own_trajectories=False):
    if left not in packs or right not in packs:
        return "one_or_both_paths_have_no_observations"
    lp, rp = packs[left], packs[right]
    empty_left = {pid for pid, path in collector.empty_records if path == lp["path"]}
    empty_right = {pid for pid, path in collector.empty_records if path == rp["path"]}
    if not own_trajectories and empty_left != empty_right:
        return "zero_observation_prompt_ids_differ_between_paths"
    if lp["prompts"] != rp["prompts"] or lp["layers"] != rp["layers"]:
        return "incomplete_matching_prompt_or_layer_ids"
    if not own_trajectories:
        try:
            assert_matched_masks(collector, lp["path"], rp["path"], position_bin=group)
        except ValueError as exc:
            return str(exc)
        # Global sets of hashes are insufficient if parameter assignment were
        # accidentally swapped across prompts; require row-level agreement.
        for pid in lp["prompts"]:
            for layer in lp["layers"]:
                lrow = lp["rows"].get((pid, lp["path"], layer, group))
                rrow = rp["rows"].get((pid, rp["path"], layer, group))
                if lrow is None and rrow is None:
                    continue
                if lrow.calibration_hash != rrow.calibration_hash:
                    return "calibration_hash_differs_at_matched_prompt_layer"
    return None


def _analyze_group(collector, resolved, group, weighting, *, iters, seed, alpha, chunk_size):
    """Evaluate each path once per prompt draw, then share draws in contrasts."""
    packs, errors, points, path_results = {}, {}, {}, {}
    for canonical, actual in resolved.items():
        try:
            pack = _prepare(collector, actual, group)
        except ValueError as exc:
            errors[canonical] = str(exc)
            continue
        packs[canonical] = pack
        point, _ = _evaluate(pack, np.ones((1, len(pack["prompts"]))), weighting,
                             collector.protection, include_layers=True)
        points[canonical] = point
    # A matching prompt universe gets exactly the same draw for every path.
    universes = sorted({tuple(pack["prompts"]) for pack in packs.values()})
    generators = {universe: np.random.default_rng(seed) for universe in universes}
    draws = {name: {metric: [] for metric in point} for name, point in points.items()}
    for offset in range(0, iters, chunk_size):
        count = min(chunk_size, iters - offset)
        weights = {universe: generators[universe].multinomial(
            len(universe), np.full(len(universe), 1 / len(universe)), size=count).astype(np.float64)
                   for universe in universes}
        for name, pack in packs.items():
            values, _ = _evaluate(pack, weights[tuple(pack["prompts"])], weighting,
                                  collector.protection, include_layers=True)
            for metric, values_array in values.items():
                draws[name][metric].append(values_array)
    for name, pack in packs.items():
        path_results[name] = {
            "point": collector.summarize(pack["path"], position_bin=group, weighting=weighting),
            "intervals": _bootstrap_intervals(points[name], draws[name], alpha),
        }
    comparisons = {}
    pairs = [("C_minus_D", "C", "D", False), ("B_minus_A", "B", "A", False)]
    if "M_NT" in resolved or "M_ST" in resolved:
        pairs.append(("M_ST_minus_M_NT_own_trajectories", "M_ST", "M_NT", True))
    for name, left, right, own in pairs:
        error = _pair_valid(collector, left, right, group, packs, own_trajectories=own)
        if error:
            comparisons[name] = {"status": "NOT_ESTIMABLE", "reason": error}
            continue
        endpoints = ("E_norm", "E_moment")
        contrast_point = {key: points[left][key] - points[right][key] for key in endpoints}
        contrast_draws = {key: [np.concatenate(draws[left][key]) - np.concatenate(draws[right][key])]
                          for key in endpoints}
        lp, rp = packs[left], packs[right]
        comparisons[name] = {
            "status": "ok", "left": resolved[left], "right": resolved[right],
            "contrast": "left_minus_right", "prompt_count": len(lp["prompts"]),
            "matched_masks": lp["mask_hash"] == rp["mask_hash"],
            "same_calibration_required": not own, "same_mask_required": not own,
            "left_mask_hash": lp["mask_hash"], "right_mask_hash": rp["mask_hash"],
            "left_calibration_hashes": lp["calibration_hashes"],
            "right_calibration_hashes": rp["calibration_hashes"],
            "metrics": _bootstrap_intervals(contrast_point, contrast_draws, alpha),
        }
    interaction = {"status": "NOT_ESTIMABLE", "reason": "four_way_common_position_group_required"}
    if group == "common" or group.startswith("common:"):
        if all(name in packs for name in ("A", "B", "C", "D")):
            failures = [_pair_valid(collector, l, r, group, packs) for l, r in (("B", "A"), ("C", "D"))]
            if any(failures):
                interaction["reason"] = next(x for x in failures if x)
            elif not _same_position_coverage([packs[name] for name in ("A", "B", "C", "D")]):
                interaction["reason"] = "four_way_decode_positions_or_prompt_coverage_differ"
            elif any(packs[name]["calibration_hashes"] != packs["A"]["calibration_hashes"]
                     for name in ("B", "C", "D")):
                interaction["reason"] = "four_paths_do_not_observe_the_same_calibration"
            else:
                def contrast(values):
                    return {key: values["C"][key] - values["D"][key] - values["B"][key] + values["A"][key]
                            for key in ("E_norm", "E_moment")}
                point = contrast(points)
                flattened = {name: {key: np.concatenate(draws[name][key]) for key in ("E_norm", "E_moment")}
                             for name in ("A", "B", "C", "D")}
                boot = {key: [value] for key, value in contrast(flattened).items()}
                interaction = {"status": "ok", "contrast": "(C-D)-(B-A)",
                               "conditional_on_common_positions": True,
                               "prompt_count": len(packs["A"]["prompts"]),
                               "metrics": _bootstrap_intervals(point, boot, alpha)}
    return {"position_group": group, "weighting": weighting, "paths": path_results,
            "unavailable_paths": errors, "comparisons": comparisons, "interaction": interaction}


def _decide_gate(full_result, metric, native_bound, state_bound, expected_prompt_ids=None):
    paths, contrasts = full_result["paths"], full_result["comparisons"]
    decision = {"status": "INCONCLUSIVE", "gate_recalibration": False,
                "primary_metric": metric, "native_match_tolerance": native_bound,
                "state_effect_tolerance": state_bound,
                "stronger_claims_established": False,
                "scope": "state-source diagnosis only; downstream/update/confirmation evidence remains required"}
    if not all(name in paths for name in ("A", "B", "C", "D")):
        return {**decision, "reason": "required_A_B_C_D_observations_missing"}
    universes = {name: paths[name]["point"]["prompt_ids"] for name in ("A", "B", "C", "D")}
    decision["prompt_universes"] = universes
    if any(ids != universes["A"] for ids in universes.values()):
        return {**decision, "reason": "four_path_complete_prompt_universes_differ"}
    if expected_prompt_ids is not None and universes["A"] != expected_prompt_ids:
        return {**decision, "reason": "four_path_prompt_universe_differs_from_expected_ids",
                "expected_prompt_ids": expected_prompt_ids}
    if contrasts.get("B_minus_A", {}).get("status") != "ok":
        return {**decision, "reason": "supporting_fixed_text_pair_not_estimable"}
    contrast = contrasts.get("C_minus_D", {})
    if contrast.get("status") != "ok":
        return {**decision, "reason": "primary_fixed_text_pair_not_estimable",
                "details": contrast.get("reason")}
    evidence = {name: paths[name]["intervals"][metric] for name in ("A", "C", "D")}
    evidence["C_minus_D"] = contrast["metrics"][metric]
    decision["evidence"] = evidence
    if any(value.get("status") != "ok" or any(value.get(key) is None for key in
           ("estimate", "ci_low", "ci_high")) for value in evidence.values()):
        return {**decision, "reason": "primary_metric_has_undefined_estimates_or_bootstrap_draws"}
    native_valid = evidence["A"]["ci_high"] <= native_bound
    decision["native_matching_established"] = native_valid
    decision["student_deployment_mismatch_established"] = evidence["C"]["ci_low"] > native_bound
    decision["state_effect_exceeds_practical_bound"] = evidence["C_minus_D"]["ci_low"] > state_bound
    if not native_valid:
        reason = ("calibration_heldout_native_not_valid" if evidence["A"]["estimate"] > native_bound
                  else "heldout_native_matching_not_established_with_current_uncertainty")
        return {**decision, "reason": reason, "H1_supported": False}
    if decision["student_deployment_mismatch_established"] and decision["state_effect_exceeds_practical_bound"]:
        return {**decision, "status": "STATE_MISMATCH_SUPPORTED", "gate_recalibration": True,
                "reason": "native_matches_student_mismatches_and_fixed_text_state_effect_exceeds_bound",
                "H1_supported": True}
    equivalent_effect = (-state_bound <= evidence["C_minus_D"]["ci_low"] and
                         evidence["C_minus_D"]["ci_high"] <= state_bound)
    if all(evidence[name]["ci_high"] <= native_bound for name in ("A", "C", "D")) and equivalent_effect:
        return {**decision, "status": "NEGATIVE", "reason": "A_C_D_match_and_state_difference_interval_within_practical_bounds",
                "H1_supported": False}
    return {**decision, "reason": "evidence_does_not_meet_predeclared_positive_or_negative_gate",
            "H1_supported": False}


def _write_summary(out_dir, summary):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "diagnostic_summary.json"
    if destination.exists():
        old = destination.read_bytes()
        digest = hashlib.sha256(old).hexdigest()[:16]
        archive = out_dir / f"diagnostic_summary.previous.{digest}.json"
        if not archive.exists():
            with archive.open("xb") as handle:
                handle.write(old)
    temporary = out_dir / f".diagnostic_summary.{os.getpid()}.tmp"
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, destination)
    return destination


def analyze_diagnostics(collector_dir, out_dir, iters=10000, family="lda",
                        native_match_tolerance=0.05, state_effect_tolerance=0.01,
                        *, seed=20261019, alpha=0.05, chunk_size=128,
                        expected_prompt_ids=None):
    """Rebuild all diagnostic tables and a conservative predeclared P3 gate.

    LDA's primary metric is E_moment (relative joint mean/std discrepancy), and
    scalar's is E_norm (mean absolute log norm ratio). The two engineering
    bounds are independently specified study conventions, not field standards
    or an LDA reuse of a 5% norm-ratio band. Point/token and equal-prompt analyses
    use the same frozen parameters; bootstrap uncertainty excludes refitting.
    """
    started = time.perf_counter()
    if family not in ("lda", "scalar"):
        raise ValueError("family must be lda or scalar")
    if (iters < 1 or chunk_size < 1 or not 0 < alpha < 1 or
        not math.isfinite(native_match_tolerance) or native_match_tolerance < 0 or
        not math.isfinite(state_effect_tolerance) or state_effect_tolerance < 0):
        raise ValueError("invalid analysis protocol")
    if expected_prompt_ids is not None:
        expected_prompt_ids = [str(pid) for pid in expected_prompt_ids]
        if not expected_prompt_ids or len(set(expected_prompt_ids)) != len(expected_prompt_ids):
            raise ValueError("expected_prompt_ids must be nonempty and unique after normalization")
        expected_prompt_ids = sorted(expected_prompt_ids)
    source = Path(collector_dir).resolve()
    collector = StateCollector.load(source)
    resolved, available = _path_aliases(collector)
    completeness = {}
    for name, actual in resolved.items():
        actual_ids = sorted({key[0] for key in collector.records if key[1] == actual} |
                            {pid for pid, path in collector.empty_records if path == actual})
        expected = set(expected_prompt_ids) if expected_prompt_ids is not None else set(actual_ids)
        missing, unexpected = sorted(expected - set(actual_ids)), sorted(set(actual_ids) - expected)
        completeness[name] = {"path": actual, "recorded_prompt_ids": actual_ids,
                              "recorded_prompt_count": len(actual_ids),
                              "expected_prompt_count": len(expected) if expected_prompt_ids is not None else None,
                              "missing_ids": missing, "unexpected_ids": unexpected,
                              "complete_expected_ids": not missing and not unexpected if expected_prompt_ids is not None else None}
    metric = "E_moment" if family == "lda" else "E_norm"
    groups = ["all"] + [f"{lo}-{hi}" for lo, hi in collector.bins]
    has_common = any(key[3] == "common" for key in collector.records)
    if has_common:
        groups += ["common"] + [f"common:{lo}-{hi}" for lo, hi in collector.bins]
    results, coverage = {}, {}
    for group in groups:
        group_started = time.perf_counter()
        coverage[group] = _coverage(collector, resolved, group)
        if not any(row["prompt_count"] for row in coverage[group].values()):
            results[group] = {"status": "NO_OBSERVATIONS", "reason": "no_valid_decode_positions_in_predeclared_group"}
            continue
        results[group] = {weighting: _analyze_group(collector, resolved, group, weighting,
                          iters=iters, seed=seed, alpha=alpha, chunk_size=chunk_size)
                          for weighting in ("token", "prompt")}
        results[group]["elapsed_seconds"] = time.perf_counter() - group_started
    full = results.get("all", {})
    if "token" in full:
        gate = _decide_gate(full["token"], metric, native_match_tolerance, state_effect_tolerance,
                            expected_prompt_ids=expected_prompt_ids)
    else:
        gate = {"status": "INCONCLUSIVE", "gate_recalibration": False,
                "reason": "no_full_trajectory_observations", "primary_metric": metric}
    if collector.incomplete_archives:
        gate = {**gate, "status": "INCONCLUSIVE", "gate_recalibration": False,
                "reason": "uncommitted_diagnostic_archives_require_resume_or_reconciliation"}
    files = sorted([*source.glob("*.npz"), *source.glob("*.complete.json"),
                    *source.glob("*.empty.json")]) if source.is_dir() else [source]
    summary = {
        "schema_version": "state_study_analysis.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source),
        "source_archives": [{"path": str(path), "size_bytes": path.stat().st_size} for path in files],
        "family": family, "chosen_mismatch_metric": metric, "resolved_paths": resolved,
        "expected_prompt_ids": expected_prompt_ids,
        "prompt_completeness": completeness,
        "available_paths": available, "missing_core_paths": [x for x in ("A", "B", "C", "D") if x not in resolved],
        "bootstrap": {"unit": "prompt", "paired": True, "resamples": iters,
                      "seed": seed, "confidence_level": 1 - alpha, "chunk_size": chunk_size,
                      "recompute_ratios_and_population_moments_each_draw": True,
                      "calibration_parameters_refitted": False,
                      "includes_empty_and_inactive_prompt_ids": True,
                      "zero_effective_token_draw_policy": "undefined; withhold CI and scientific gate, never discard draw",
                      "uncertainty_scope": "evaluation prompt sampling conditional on frozen calibration"},
        "decision_protocol": {"native_match_tolerance": native_match_tolerance,
                              "state_effect_tolerance": state_effect_tolerance,
                              "bounds_origin": "predeclared study-specific engineering bounds; no validation tuning",
                              "primary_metric_units": "relative joint mean/std discrepancy" if family == "lda" else
                                  "mean absolute log norm ratio",
                              "student_mismatch_rule": "C primary-metric CI lower > native_match_tolerance",
                              "primary_analysis_weighting": "token", "sensitivity_weighting": "prompt"},
        "coverage": coverage, "analyses": results, "decision": gate,
        "gate_recalibration": gate["gate_recalibration"], "status": gate["status"],
        "common_prefix_analysis": {"available": has_common,
                                   "mask_predeclared_during_collection": has_common,
                                   "scope": "conditional common positions; full long suffix remains in primary analysis"},
        "zero_observation_markers": [{"prompt_id": pid, "path": path, "count": 0, "reason": reason,
                                      "metadata": collector.empty_metadata.get((pid, path), {})}
                                     for (pid, path), reason in sorted(collector.empty_records.items())],
        "incomplete_uncommitted_archives": collector.incomplete_archives,
        "limitations": ["Zero-decode and inactive-bin prompts remain in the bootstrap universe with zero counts; they carry no invented moments. Draws without states are explicitly undefined.",
                        "Missing position bins are unobserved, not evidence of matching.",
                        "STATE_MISMATCH_SUPPORTED does not establish updated-model or downstream benefit.",
                        "No equivalence claim follows from an interval merely including zero."],
        "runtime": {"elapsed_seconds": time.perf_counter() - started, "device": "cpu",
                    "blas_thread_environment": {key: os.environ.get(key) for key in
                        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
    }
    _write_summary(out_dir, summary)
    return summary
