"""Manifest + config-hash helpers for reproducibility (P0 requirement #10)."""
import json, hashlib, os, subprocess, time


def git_commit(repo):
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def write_manifest(out_dir, *, tag, config, model, env, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    repo = "/home/t-jialianggu/work/MOEresearch"
    manifest = {
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(repo),
        "model": model,
        "config": config,
        "config_hash": config_hash(config),
        "env": env,
    }
    if extra:
        manifest.update(extra)
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    return manifest
