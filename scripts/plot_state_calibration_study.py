#!/usr/bin/env python3
"""Export up to three figure types from measured state-study summaries.

No model is loaded, no results are synthesized, and scopes/caps are never
pooled. Unavailable figures or panels are listed in plot_manifest.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ARM_ORDER = ("N", "F", "R8", "R16", "M_N", "L_N", "M_NT", "M_ST")
COLORS = {arm: color for arm, color in zip(ARM_ORDER, plt.get_cmap("tab10").colors)}
PATH_LABELS = {"A": "A: Native driver / Native text", "B": "B: Student driver / Native text",
               "C": "C: Student driver / Student text", "D": "D: Native driver / Student text"}


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _has_summary(scope):
    return any((scope / path).is_file() for path in (
        "summaries/diagnostic_summary.json", "summaries/generation_summary.json",
        "select/summaries/generation_summary.json", "validate/summaries/generation_summary.json"))


def select_inputs(run_dir, scope="auto", generation_role="auto"):
    run_dir = Path(run_dir).resolve()
    if scope != "auto":
        selected = run_dir if run_dir.name == scope else run_dir / scope
    elif _has_summary(run_dir):
        selected = run_dir
    else:
        selected = next((run_dir / name for name in ("formal", "pilot", "smoke")
                         if _has_summary(run_dir / name)), run_dir)
    diagnostic = selected / "summaries/diagnostic_summary.json"
    diagnostic = diagnostic if diagnostic.is_file() else None
    candidates = ([("validate", selected / "validate/summaries/generation_summary.json"),
                   ("select", selected / "select/summaries/generation_summary.json"),
                   (selected.name, selected / "summaries/generation_summary.json")]
                  if generation_role == "auto" else
                  [(generation_role, selected / generation_role / "summaries/generation_summary.json")])
    generation = next(((role, path) for role, path in candidates if path.is_file()), (None, None))
    return selected, diagnostic, generation[1], generation[0]


def _load(path):
    if path is None:
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _layer_series(path_result, metric):
    if not path_result:
        return None
    intervals = path_result.get("intervals", {})
    points = path_result.get("point", {}).get("layers", [])
    layers = sorted(row["layer"] for row in points)
    if not layers:
        return None
    values, lows, highs = [], [], []
    for layer in layers:
        interval = intervals.get(f"layer:{layer}:{metric}", {})
        valid = interval.get("status") == "ok" and all(
            finite(interval.get(key)) for key in ("estimate", "ci_low", "ci_high"))
        values.append(interval["estimate"] if valid else np.nan)
        lows.append(interval["ci_low"] if valid else np.nan)
        highs.append(interval["ci_high"] if valid else np.nan)
    if not np.isfinite(values).any():
        return None
    return np.asarray(layers), np.asarray(values), np.asarray(lows), np.asarray(highs)


def _style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=.18, linewidth=.6)
    axis.set_axisbelow(True)


def _export(figure, out_dir, name, dpi, description, omitted_panels):
    paths = []
    for suffix in ("pdf", "png"):
        destination = out_dir / f"{name}.{suffix}"
        figure.savefig(destination, dpi=dpi, bbox_inches="tight")
        paths.append(str(destination))
    plt.close(figure)
    return {"name": name, "files": paths, "description": description,
            "omitted_panels": omitted_panels}


def plot_state_mismatch(diagnostic, out_dir, stem, dpi, *, scope_name=None):
    paths = diagnostic.get("analyses", {}).get("all", {}).get("token", {}).get("paths", {})
    pairs = [("Native text (A/B)", ("A", "B")), ("Student text (D/C)", ("D", "C"))]
    usable = [(title, pair) for title, pair in pairs if any(
        _layer_series(paths.get(path), metric) is not None for path in pair
        for metric in ("signed_log_ratio", "E_moment"))]
    if not usable:
        return None, "No measured per-layer diagnostic intervals are available."
    figure, axes = plt.subplots(2, len(usable), figsize=(6.1 * len(usable), 7.1), squeeze=False,
                               sharex="col", constrained_layout=True)
    omitted = []
    confidence = 100 * diagnostic.get("bootstrap", {}).get("confidence_level", .95)
    for column, (title, pair) in enumerate(usable):
        for row, metric in enumerate(("signed_log_ratio", "E_moment")):
            axis = axes[row, column]
            observed = False
            for path in pair:
                series = _layer_series(paths.get(path), metric)
                if series is None:
                    omitted.append(f"{title}: {path} {metric} unavailable")
                    continue
                layers, estimates, low, high = series
                color = "#2673ad" if path in ("A", "D") else "#d86a28"
                axis.plot(layers, estimates, color=color, linewidth=1.4, label=PATH_LABELS[path])
                axis.fill_between(layers, low, high, color=color, alpha=.18, linewidth=0)
                observed = True
            _style(axis)
            if metric == "signed_log_ratio":
                axis.axhline(0, color=".35", linestyle="--", linewidth=.8)
                axis.set_ylabel("Signed log norm ratio")
            else:
                axis.set_ylabel("Relative joint mean/std discrepancy")
                axis.set_ylim(bottom=0)
            if observed:
                axis.legend(fontsize=8, frameon=False)
            else:
                axis.set_visible(False)
                omitted.append(f"{title}: entire {metric} panel omitted")
            if row == 0:
                axis.set_title(title)
            if row == 1:
                axis.set_xlabel("MoE layer (zero-based)")
    counts = sorted({value.get("point", {}).get("prompt_count") for value in paths.values()
                     if value.get("point", {}).get("prompt_count") is not None})
    context = f"{scope_name}: " if scope_name else ""
    context += "engineering check; " if scope_name in ("smoke", "pilot") else ""
    context += f"n={','.join(map(str, counts))} prompts"
    draws = diagnostic.get("bootstrap", {}).get("resamples")
    figure.suptitle(f"Same-text state comparison ({context})\n"
                   f"Full observed decode trace; token weighting; {confidence:g}% prompt-bootstrap intervals ({draws} resamples)", fontsize=11)
    return _export(figure, out_dir, f"01_state_mismatch_{stem}", dpi,
                   "A/B and D/C use their own fixed texts; bands condition on frozen calibration. "
                   "The two text-source panels can cover different trace lengths.", omitted), None


def _categorical_panel(axis, labels, values, ylabel, *, percent=False, intervals=None):
    positions = np.arange(len(labels))
    for i, (label, value) in enumerate(zip(labels, values)):
        color = COLORS.get(label, ".3")
        multiplier = 100 if percent else 1
        axis.scatter(i, value * multiplier, color=color, s=58, zorder=3)
        if intervals and intervals[i] is not None:
            lo, hi = intervals[i]
            # Draw the interval separately so a percentile interval need not
            # contain the point estimate to be rendered correctly.
            axis.vlines(i, lo * multiplier, hi * multiplier, color=color, linewidth=1.7)
            axis.plot([i - .06, i + .06], [lo * multiplier] * 2, color=color, linewidth=1.3)
            axis.plot([i - .06, i + .06], [hi * multiplier] * 2, color=color, linewidth=1.3)
    axis.set_xticks(positions, labels)
    axis.set_ylabel(ylabel)
    if percent:
        axis.set_ylim(-2, 102)
    else:
        axis.set_ylim(bottom=0)
    axis.set_xlim(-.5, len(labels) - .5)
    _style(axis)


def plot_updated(diagnostic, generation, role, out_dir, stem, dpi):
    paths = (diagnostic or {}).get("analyses", {}).get("all", {}).get("token", {}).get("paths", {})
    arms = (generation or {}).get("arms", {})
    if not (any(path in paths for path in ("M_NT", "M_ST")) or any(arm in arms for arm in ("M_NT", "M_ST"))):
        return None, "Updated models have not produced own-trajectory or generation summaries."
    metric = (diagnostic or {}).get("chosen_mismatch_metric", "E_moment")
    panels, omitted = [], []
    labels, values, intervals = [], [], []
    for label, path in (("L_N", "C"), ("M_NT", "M_NT"), ("M_ST", "M_ST")):
        interval = paths.get(path, {}).get("intervals", {}).get(metric, {})
        if interval.get("status") == "ok" and all(finite(interval.get(key)) for key in ("estimate", "ci_low", "ci_high")):
            labels.append(label); values.append(interval["estimate"])
            intervals.append((interval["ci_low"], interval["ci_high"]))
    if labels:
        panels.append(("Own trajectories: C_diag", labels, values, metric, False, intervals))
    else:
        omitted.append("C_diag own-trajectory mismatch: not run or unavailable")
    population = "D_validate" if role == "validate" else "D_select" if role == "select" else role or "recorded requests"
    for field, label, percent in (("eos_rate", "Natural EOS (%)", True),
                                  ("mean_generated_tokens", "Observed output tokens/request", False)):
        labels, values = [], []
        for arm in ("L_N", "M_NT", "M_ST"):
            value = arms.get(arm, {}).get(field)
            if finite(value):
                labels.append(arm); values.append(value)
        if labels:
            panels.append((f"Free generation: {population}", labels, values, label, percent, None))
        else:
            omitted.append(f"{population} {field}: not run or unavailable")
    if not panels:
        return None, "Updated summaries contain no finite mismatch, EOS, or length measurements."
    figure, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 4), squeeze=False,
                               constrained_layout=True)
    for axis, (title, labels, values, ylabel, percent, intervals) in zip(axes[0], panels):
        _categorical_panel(axis, labels, values, ylabel, percent=percent, intervals=intervals)
        axis.set_title(title, fontsize=10)
    figure.suptitle("Initial and updated models on their own trajectories", fontsize=12)
    return _export(figure, out_dir, f"02_updated_own_trajectories_{stem}", dpi,
                   "Mismatch bands are prompt-bootstrap intervals on C_diag; generation panels are separate "
                   f"{population} point estimates. EOS/cap/request failures remain represented in source summaries. "
                   "Capped observed lengths do not identify natural lengths beyond the cap.", omitted), None


def plot_quality_cost(generation, role, out_dir, stem, dpi, *, scope_name=None):
    arms = generation.get("arms", {})
    specs = [("mean_generated_tokens", "Observed output tokens/request\n(EOS/cap/failure when measured)"),
             ("mean_generation_seconds", "Observer-free generation seconds/request\n(current backend)")]
    panels, omitted = [], []
    for field, label in specs:
        available = [arm for arm in ARM_ORDER if arm in arms and finite(arms[arm].get(field)) and
                     any(finite(arms[arm].get(score)) for score in ("strict_accuracy", "tolerant_accuracy"))]
        if available:
            panels.append((field, label, available))
        else:
            omitted.append(f"{field}: no finite scored request measurements")
    if not panels:
        return None, "No paired quality/cost point estimates are available."
    figure, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.8), squeeze=False,
                               constrained_layout=True)
    for axis, (field, xlabel, available) in zip(axes[0], panels):
        annotation_groups = {}
        for arm in available:
            data = arms[arm]
            x = data[field]
            alpha = 1 if data.get("complete_expected_ids") else .5
            scores = []
            for score, marker in (("strict_accuracy", "o"), ("tolerant_accuracy", "s")):
                value = data.get(score)
                if not finite(value):
                    continue
                y = value * 100
                scores.append(y)
                axis.scatter(x, y, marker=marker, s=57, edgecolors=COLORS[arm],
                             facecolors=COLORS[arm] if marker == "o" else "none", alpha=alpha,
                             linewidth=1.2, zorder=3)
            if len(scores) == 2:
                axis.plot([x, x], scores, color=COLORS[arm], alpha=alpha * .5, linewidth=1)
            if scores:
                annotation_groups.setdefault((x, max(scores)), []).append(arm)
        for index, ((x, y), coincident) in enumerate(sorted(annotation_groups.items())):
            label = "\n".join(", ".join(coincident[start:start + 3]) for start in range(0, len(coincident), 3))
            color = COLORS[coincident[0]] if len(coincident) == 1 else ".25"
            offset = (11 + 13 * (index % 3)) * (-1 if y >= 90 else 1)
            axis.annotate(label, (x, y), xytext=(6, offset), textcoords="offset points",
                          fontsize=8, color=color, va="top" if offset < 0 else "bottom",
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": .75, "pad": .5},
                          arrowprops={"arrowstyle": "-", "linewidth": .55, "color": color})
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Correct requests (%)")
        axis.set_ylim(-2, 102)
        _style(axis)
        axis.legend(handles=[Line2D([], [], color=".25", marker="o", linestyle="none", label="Strict"),
                             Line2D([], [], color=".25", marker="s", markerfacecolor="none",
                                    linestyle="none", label="Tolerant")], frameon=False, fontsize=9)
    missing = any(not arms[arm].get("complete_expected_ids") for _, _, available in panels for arm in available)
    caps = sorted({cap for data in arms.values() for cap in data.get("max_new_tokens_values", [])})
    counts = sorted({data.get("n_recorded") for data in arms.values() if data.get("n_recorded") is not None})
    context = f"{scope_name + ' / ' if scope_name else ''}{role or 'recorded requests'}"
    if scope_name in ("smoke", "pilot"):
        context += " (engineering check)"
    context += f"; cap={','.join(map(str, caps))}; n/arm={','.join(map(str, counts))}"
    figure.suptitle(f"Quality and request cost: {context}\n"
                   "Point estimates; strict and tolerant scores shown together" +
                   ("; faded = incomplete planned IDs" if missing else ""), fontsize=12)
    return _export(figure, out_dir, f"03_quality_request_cost_{stem}", dpi,
                   "No arm-level confidence intervals are invented from paired-difference CIs. "
                   "Accuracy retains request failures as specified by the scoring summary; available cost "
                   "measurements have explicit counts. Administrative caps censor natural output length. "
                   "Timing excludes observers and debug-logit runs and applies only to the measured backend.", omitted), None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="One run or one explicit formal/pilot/smoke scope")
    parser.add_argument("--out-dir", type=Path, required=True, help="Scratch directory for PDF, PNG and manifest")
    parser.add_argument("--scope", choices=("auto", "formal", "pilot", "smoke"), default="auto")
    parser.add_argument("--generation-role", choices=("auto", "validate", "select"), default="auto")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    selected, diagnostic_path, generation_path, role = select_inputs(args.run_dir, args.scope, args.generation_role)
    inputs = [path for path in (diagnostic_path, generation_path) if path is not None]
    source_info = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in inputs]
    digest = hashlib.sha256(json.dumps(source_info, sort_keys=True).encode()).hexdigest()[:10]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    diagnostic, generation = _load(diagnostic_path), _load(generation_path)
    stem = f"{selected.name}_{digest}"
    manifest = {"schema_version": "state_study_figures.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(args.run_dir.resolve()), "selected_scope": str(selected), "generation_role": role,
                "source_summaries": source_info, "figures": [], "skipped_figures": [],
                "matplotlib_version": matplotlib.__version__, "scope_pooling": False,
                "data_policy": "Measured summaries only; missing panels omitted; no synthetic values or intervals."}
    jobs = [("state_mismatch", lambda: plot_state_mismatch(diagnostic, args.out_dir, stem, args.dpi,
                                                         scope_name=selected.name), diagnostic is not None),
            ("updated_own_trajectories", lambda: plot_updated(diagnostic, generation, role, args.out_dir, stem, args.dpi),
             diagnostic is not None or generation is not None),
            ("quality_request_cost", lambda: plot_quality_cost(generation, role, args.out_dir, stem, args.dpi,
                                                              scope_name=selected.name), generation is not None)]
    for name, function, available in jobs:
        figure, reason = function() if available else (None, "Required measured summary is unavailable.")
        if figure is not None:
            manifest["figures"].append(figure)
        else:
            manifest["skipped_figures"].append({"name": name, "reason": reason})
    manifest["status"] = "GENERATED" if manifest["figures"] else "NO_MEASURED_SUMMARIES_TO_PLOT"
    destination = args.out_dir / "plot_manifest.json"
    if destination.exists():
        old = destination.read_bytes()
        backup = args.out_dir / f"plot_manifest.previous.{hashlib.sha256(old).hexdigest()[:12]}.json"
        if not backup.exists():
            backup.write_bytes(old)
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": manifest["status"], "figure_count": len(manifest["figures"]),
                      "manifest": str(destination), "selected_scope": str(selected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
