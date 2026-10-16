"""Data prep: download raw shards, tensorize them to Drive once (CPU runtime), then load compact tensors into RAM.
Split by shard id % 10: 8 = val, 9 = test, everything else train."""

import json
import os
import random
import subprocess
import time

import numpy as np
import torch

from config import (AGENT_FEAT_DIM, DOWNLOAD_BATCH, DOWNLOAD_RETRIES, GCS_PREFIX, HIST_STEPS, LOCAL_SHARD_DIR,
                    MAX_AGENTS, MAX_STEP_SPEED_MPS, PREP_OUT_DIR, PREP_SHARD_IDS, SEED, SHARDS_PER_CHUNK)
from data import build_model_inputs, collate_scenes, extract_scene_agents, load_dataset, target_type


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def remote_shard(i): return f"{GCS_PREFIX}/training_tfexample.tfrecord-{i:05d}-of-01000"
def local_shard(i):  return f"{LOCAL_SHARD_DIR}/training_tfexample.tfrecord-{i:05d}-of-01000"


def download_shards(ids):
    """gsutil-copies the shards not already on local disk (needs `gcloud auth login`)."""
    need = [i for i in ids if not os.path.exists(local_shard(i))]
    if need:
        subprocess.run(["gsutil", "-m", "cp", *[remote_shard(i) for i in need], LOCAL_SHARD_DIR + "/"], check=False)


def shard_split(shard_id):
    r = shard_id % 10
    return "test" if r == 9 else ("val" if r == 8 else "train")


def split_shard_counts(shard_ids=PREP_SHARD_IDS):
    """(train, val, test) shard counts."""
    n_val = sum(1 for i in shard_ids if i % 10 == 8)
    n_test = sum(1 for i in shard_ids if i % 10 == 9)
    return len(shard_ids) - n_val - n_test, n_val, n_test


def _velocity_ok(inp):
    # Drop physically impossible tracks (implied speed between consecutive valid future steps above the threshold)
    fx = inp["future_xy"].numpy()
    fv = inp["future_valid"].numpy() > 0
    step = np.linalg.norm(np.diff(fx, axis=0), axis=1)          # [T-1] metres between adjacent steps
    both = fv[:-1] & fv[1:]
    return not (both.any() and (step[both] / 0.1 > MAX_STEP_SPEED_MPS).any())


def _pack(s):   # compact storage: fp16 features, fp32 targets, uint8 masks, int16 road types
    return {"agent_feats": s["agent_feats"].half(),
            "agent_valid": (s["agent_valid"] > 0).to(torch.uint8),
            "future_xy": s["future_xy"].float(),
            "future_valid": (s["future_valid"] > 0).to(torch.uint8),
            "road_xy": s["road_xy"].half(),
            "road_type": s["road_type"].to(torch.int16),
            "road_valid": (s["road_valid"] > 0).to(torch.uint8)}


def run_prep(out_dir=PREP_OUT_DIR, shard_ids=PREP_SHARD_IDS):
    """Downloads, tensorizes and saves shards in chunks of SHARDS_PER_CHUNK. Resumable: finished chunks are skipped."""
    os.makedirs(out_dir, exist_ok=True)
    n_chunks = (len(shard_ids) + SHARDS_PER_CHUNK - 1) // SHARDS_PER_CHUNK
    incomplete = []
    shape_checked = False
    for cidx in range(n_chunks):
        shard_batch = shard_ids[cidx * SHARDS_PER_CHUNK:(cidx + 1) * SHARDS_PER_CHUNK]
        marker = os.path.join(out_dir, f"chunk_{cidx:03d}_done")
        if _read_json(marker) == shard_batch:   # done only if built from exactly these shards
            print(f"chunk {cidx}: already saved, skipping"); continue
        buf = {"train": [], "val": [], "test": []}
        n_dropped, missing = 0, []
        t0 = time.time()
        for b0 in range(0, len(shard_batch), DOWNLOAD_BATCH):
            ids = shard_batch[b0:b0 + DOWNLOAD_BATCH]
            for _attempt in range(DOWNLOAD_RETRIES):
                if all(os.path.exists(local_shard(i)) for i in ids):
                    break
                download_shards(ids)
            for i in ids:
                if not os.path.exists(local_shard(i)):
                    missing.append(i); continue
                split = shard_split(i)
                for parsed in load_dataset(local_shard(i)):
                    for scene in extract_scene_agents(parsed, max_targets=10_000):
                        inp = build_model_inputs(scene)
                        if not shape_checked:
                            # Guard against stale settings: stored shapes must match config.py
                            want = (MAX_AGENTS, HIST_STEPS, AGENT_FEAT_DIM)
                            if tuple(inp["agent_feats"].shape) != want:
                                raise RuntimeError(f"agent_feats is {tuple(inp['agent_feats'].shape)}, expected {want}.")
                            shape_checked = True
                        if inp["future_valid"].sum() == 0:
                            continue
                        if not _velocity_ok(inp):
                            n_dropped += 1
                            continue
                        buf[split].append(_pack(inp))
                os.remove(local_shard(i))
        counts = {}
        for split in ("train", "val", "test"):
            fpath = os.path.join(out_dir, f"chunk_{cidx:03d}_{split}.pt")
            if buf[split]:
                stacked = {k: torch.stack([smp[k] for smp in buf[split]]) for k in buf[split][0]}
                torch.save(stacked, fpath)
                counts[split] = len(buf[split])
            elif os.path.exists(fpath):
                os.remove(fpath)   # stale file from an earlier partial run
        if missing:
            incomplete.append(cidx)
            print(f"chunk {cidx}: {counts} -- NOT marked done, failed to download shards {missing}")
        else:
            _write_json(marker, shard_batch)
            print(f"chunk {cidx}: {counts}, dropped {n_dropped} by speed check, "
                  f"in {(time.time() - t0) / 60:.1f} min")
    if incomplete:
        print(f"INCOMPLETE: chunks {incomplete} are missing shards. Check gsutil auth "
              f"(gcloud auth login), then run prep again -- finished chunks are skipped.")
    else:
        _write_json(os.path.join(out_dir, "PREP_DONE"), shard_ids)
        print("PREP COMPLETE.")


def prep_done(out_dir=PREP_OUT_DIR, shard_ids=PREP_SHARD_IDS):
    """True if out_dir holds a finished prep of exactly these shards."""
    done = _read_json(os.path.join(out_dir, "PREP_DONE"))
    if done is None:
        return False
    if done != shard_ids:
        raise RuntimeError(f"{out_dir} was not built from the current PREP_SHARD_IDS "
                           f"({len(shard_ids)} shards). Delete that folder in Drive and prep again.")
    return True


def load_split(split, out_dir=PREP_OUT_DIR):
    """All saved chunks of one split -> list of per-sample dicts (views into the compact tensors)."""
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(f"_{split}.pt"))
    if not files:
        print(f"WARNING: no '{split}' files in {out_dir} -- this split is empty.")
        return []
    acc = {}
    t0 = time.time()
    for j, f in enumerate(files):
        st = torch.load(os.path.join(out_dir, f))
        for k, v in st.items():
            acc.setdefault(k, []).append(v)
        if (j + 1) % 5 == 0 or j + 1 == len(files):
            print(f"  {split}: read {j + 1}/{len(files)} files ({(time.time() - t0) / 60:.1f} min)", flush=True)
    acc = {k: torch.cat(v) for k, v in acc.items()}
    h = torch.tensor([1.0, 0.0])
    n = len(acc["agent_feats"])
    # Views into the compact tensors; collate_scenes casts each batch
    return [{k: acc[k][i] for k in acc} | {"heading": h} for i in range(n)]


def load_splits(splits=("train", "val", "test"), out_dir=PREP_OUT_DIR):
    """Loads the requested splits (others come back empty). train is shuffled with SEED, so anchors and
    mode indices are reproducible across runs."""
    t0 = time.time()
    out = {s: (load_split(s, out_dir) if s in splits else []) for s in ("train", "val", "test")}
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    random.shuffle(out["train"])
    print(f"loaded train {len(out['train'])}, val {len(out['val'])}, test {len(out['test'])} "
          f"in {(time.time() - t0) / 60:.1f} min")
    return out["train"], out["val"], out["test"]


def describe_splits(train_scenes, val_scenes):
    """Target-type mix, far-endpoint share and stored shapes / dtypes."""
    for name, split in (("train", train_scenes), ("val", val_scenes)):
        if split:
            ty = np.array([target_type(s) for s in split[:5000]])
            print(f"{name} target types: vehicle {np.mean(ty == 0):.1%}, pedestrian {np.mean(ty == 1):.1%}, "
                  f"cyclist {np.mean(ty == 2):.1%}")
    if train_scenes:
        # Share of samples whose last valid point is beyond 150 m
        end = []
        for s in train_scenes[:20000]:
            v = np.where(s["future_valid"].numpy() > 0)[0]
            end.append(float(np.linalg.norm(s["future_xy"][v[-1]].numpy())))
        end = np.array(end)
        print(f"train endpoints beyond 150 m: {np.mean(end > 150):.2%}, "
              f"max {end.max():.0f} m")
        s = train_scenes[0]
        print("sample shapes/dtypes:", {k: (tuple(v.shape), str(v.dtype).replace("torch.", "")) for k, v in s.items()})
        b = collate_scenes(train_scenes[:4])
        print("collated batch dtypes:", [str(t.dtype).replace("torch.", "") for t in b])
