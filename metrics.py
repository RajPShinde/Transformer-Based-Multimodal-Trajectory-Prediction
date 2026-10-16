"""Metrics: top-6 (endpoint NMS) minADE / minFDE / miss rate, calibration, off-road rate, Waymo-style metrics,
and the printed evaluation report."""

import numpy as np
import torch
import torch.nn.functional as F

from config import (EVAL_TOP_K, FUTURE_STEPS, MAX_ROAD_POINTS, MISS_THRESHOLD_M, NMS_DIST_M, NUM_MODES,
                    OFFROAD_STEP_STRIDE, OFFROAD_THRESH_M, POS_SCALE_M, ROAD_FORWARD_OFFSET_M, VELOCITY_SCALE_MPS)
from data import collate_scenes, target_type

TYPE_SLOT_NAMES = ["vehicle", "pedestrian", "cyclist"]


def _device(model):
    return next(model.parameters()).device


def select_modes(pred_trajs, pred_confs, k, dist_thresh=NMS_DIST_M):
    """pred_trajs [B, M, T, 2], pred_confs [B, M] -> (idx [B, k], probs [B, k]), most confident first.
    Endpoint NMS: keep a mode unless its final point is within dist_thresh of an already-kept mode's
    final point; if fewer than k survive, fill with the most confident suppressed modes.
    idx[:, 0] is always the single most confident mode."""
    B, M = pred_confs.shape
    k = min(k, M)
    dev = pred_confs.device
    probs = F.softmax(pred_confs, dim=-1)                                              # [B, M]
    order = probs.argsort(dim=1, descending=True)                                      # [B, M]
    ends = torch.gather(pred_trajs[:, :, -1], 1, order.unsqueeze(-1).expand(B, M, 2))  # [B, M, 2], sorted
    close = torch.cdist(ends, ends) < dist_thresh                                      # [B, M, M]
    kept = torch.zeros(B, M, dtype=torch.bool, device=dev)
    n_kept = torch.zeros(B, dtype=torch.long, device=dev)
    for j in range(M):
        suppressed = (close[:, j] & kept).any(dim=1)
        take = ~suppressed & (n_kept < k)
        kept[:, j] = take
        n_kept += take.long()
    rank_key = (~kept).long() * M + torch.arange(M, device=dev)    # kept modes first, each group by confidence
    pos = rank_key.argsort(dim=1)[:, :k]
    idx = torch.gather(order, 1, pos)
    return idx, torch.gather(probs, 1, idx)


def _offroad_flags(traj, traj_valid, lane_xy, lane_mask, reach, center):
    """traj [B, K, S, 2] m, traj_valid [B, K, S] bool, lane_xy [B, R, 2] m, lane_mask [B, R] bool,
    reach [B] m -> [B, K] bool: any checkable point farther than the threshold from every lane point."""
    B, K, S, _ = traj.shape
    d = torch.cdist(traj.reshape(B, K * S, 2), lane_xy)                       # [B, K*S, R]
    d = d.masked_fill(~lane_mask.unsqueeze(1), float("inf")).min(dim=-1).values.view(B, K, S)
    checkable = traj_valid & ((traj - center).norm(dim=-1) <= reach.view(B, 1, 1))
    return ((d > OFFROAD_THRESH_M) & checkable).any(dim=-1)


def compute_metrics(model, scenes, batch_size=128, miss_threshold_m=MISS_THRESHOLD_M, return_per_sample=False,
                    top_k=None):
    top_k = EVAL_TOP_K if top_k is None else top_k
    device = _device(model)
    model.eval()
    min_ades, min_fdes, misses, p_top1s, top1_best = [], [], [], [], []
    off_modes, off_top1, off_gt = [], [], []
    center = torch.tensor([ROAD_FORWARD_OFFSET_M, 0.0], device=device)   # crop center in the agent frame

    with torch.no_grad():
        for i in range(0, len(scenes), batch_size):
            batch = scenes[i : i + batch_size]
            feats, valid, future_xy, future_valid, road_xy, road_type, road_valid, heading = collate_scenes(batch)
            feats, valid = feats.to(device), valid.to(device)
            future_xy, future_valid = future_xy.to(device), future_valid.to(device)
            road_xy, road_type, road_valid = road_xy.to(device), road_type.to(device), road_valid.to(device)
            heading = heading.to(device)

            pred_trajs, pred_confs, _ = model(feats, valid, road_xy, road_type, road_valid, heading)  # [B, M, T, 2]
            B, M, T, _ = pred_trajs.shape
            k = min(top_k, M)
            top_idx, top_p = select_modes(pred_trajs, pred_confs, k)       # [B, k], most confident first
            pred_trajs = torch.gather(pred_trajs, 1, top_idx[:, :, None, None].expand(B, k, T, 2))  # [B, k, T, 2]

            dist = (pred_trajs - future_xy.unsqueeze(1)).norm(dim=-1)  # [B, k, T]
            valid_exp = future_valid.unsqueeze(1)                      # [B, 1, T]

            ade_per_mode = (dist * valid_exp).sum(-1) / valid_exp.sum(-1).clamp(min=1.0)  # [B, k]
            min_ade = ade_per_mode.min(dim=1).values

            # last valid step (validity is not always a contiguous prefix)
            time_idx = torch.arange(future_valid.shape[1], device=future_valid.device)
            last_idx = (future_valid * time_idx.unsqueeze(0)).max(dim=1).values.long()
            fde_per_mode = dist[torch.arange(B, device=dist.device), :, last_idx]  # [B, k]
            min_fde = fde_per_mode.min(dim=1).values

            min_ades.extend(min_ade.cpu().tolist())
            min_fdes.extend(min_fde.cpu().tolist())
            misses.extend((min_fde > miss_threshold_m).float().cpu().tolist())
            p_top1s.extend(top_p[:, 0].cpu().tolist())
            top1_best.extend((ade_per_mode.argmin(dim=1) == 0).float().cpu().tolist())

            # Off-road check on vehicle targets with lane centers in the crop
            map_xy = road_xy[:, :MAX_ROAD_POINTS, :2] * POS_SCALE_M                  # [B, R, 2] m
            map_valid = road_valid[:, :MAX_ROAD_POINTS] > 0
            lane_mask = map_valid & (road_type[:, :MAX_ROAD_POINTS] == 1)            # bucket 1 = lane center
            reach = torch.where(map_valid, (map_xy - center).norm(dim=-1),
                                torch.zeros_like(map_valid, dtype=map_xy.dtype)).max(dim=1).values  # [B]
            eligible = (feats[:, 0, -1, 6] > 0) & lane_mask.any(dim=1)               # vehicles with lanes

            sidx = torch.arange(OFFROAD_STEP_STRIDE - 1, T, OFFROAD_STEP_STRIDE, device=device)
            pred_s = pred_trajs[:, :, sidx]                                          # [B, k, S, 2]
            pred_off = _offroad_flags(pred_s, torch.ones(pred_s.shape[:3], dtype=torch.bool, device=device),
                                      map_xy, lane_mask, reach, center)              # [B, k]
            gt_s = future_xy[:, sidx].unsqueeze(1)                                   # [B, 1, S, 2]
            gt_off = _offroad_flags(gt_s, (future_valid[:, sidx] > 0).unsqueeze(1),
                                    map_xy, lane_mask, reach, center)[:, 0]          # [B]

            off_modes.extend(pred_off.float().mean(dim=1)[eligible].cpu().tolist())
            off_top1.extend(pred_off[:, 0].float()[eligible].cpu().tolist())
            off_gt.extend(gt_off.float()[eligible].cpu().tolist())

    model.train()
    result = {
        "minADE": float(np.mean(min_ades)),
        "minFDE": float(np.mean(min_fdes)),
        "miss_rate": float(np.mean(misses)),
        "p_top1": float(np.mean(p_top1s)),
        "top1_is_best": float(np.mean(top1_best)),
        "offroad_rate": float(np.mean(off_modes)) if off_modes else float("nan"),       # all selected modes
        "offroad_top1": float(np.mean(off_top1)) if off_top1 else float("nan"),         # most confident mode
        "gt_offroad_rate": float(np.mean(off_gt)) if off_gt else float("nan"),          # baseline
    }
    if return_per_sample:
        result["per_sample"] = {"minADE": np.array(min_ades), "minFDE": np.array(min_fdes),
                                "miss": np.array(misses)}
    return result


# Waymo-style metrics: 3 / 5 / 8 s on 2 Hz points, speed-scaled lateral / longitudinal miss boxes (no mAP)

OFFICIAL_HORIZONS = ((3, 29, 1.0, 2.0), (5, 49, 1.8, 3.6), (8, 79, 3.0, 6.0))   # (s, step idx, lat, lon)
OFFICIAL_STEPS_2HZ = torch.arange(4, FUTURE_STEPS, 5)                           # 0.5 s, 1.0 s, ... 8.0 s


def _gt_heading(gt, gv, e, lag=5):
    """Unit vector of the ground truth's direction of travel over [e - lag, e]; +x if it barely moved."""
    s = max(e - lag, 0)
    d = gt[:, e] - gt[:, s]                                                   # [B, 2]
    ok = (gv[:, e] > 0) & (gv[:, s] > 0) & (d.norm(dim=-1) > 0.5)
    fallback = torch.tensor([1.0, 0.0], device=gt.device).expand_as(d)
    return torch.where(ok.unsqueeze(-1), d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6), fallback)


def compute_official_metrics(model, scenes, batch_size=256, top_k=None):
    top_k = EVAL_TOP_K if top_k is None else top_k
    device = _device(model)
    model.eval()
    acc = {f"{n}@{h}s": [] for h, *_ in OFFICIAL_HORIZONS for n in ("minADE", "minFDE", "MR")}
    types = []
    steps2 = OFFICIAL_STEPS_2HZ.to(device)
    with torch.no_grad():
        for i in range(0, len(scenes), batch_size):
            feats, valid, gt, gv, road_xy, road_type, road_valid, heading = \
                [t.to(device) for t in collate_scenes(scenes[i:i + batch_size])]
            pred, conf, _ = model(feats, valid, road_xy, road_type, road_valid, heading)
            B, M, T, _ = pred.shape
            k = min(top_k, M)
            idx, _ = select_modes(pred, conf, k)
            pred = torch.gather(pred, 1, idx[:, :, None, None].expand(B, k, T, 2))   # [B, k, T, 2]
            err = pred - gt.unsqueeze(1)                                            # [B, k, T, 2]
            dist = err.norm(dim=-1)                                                 # [B, k, T]
            speed0 = feats[:, 0, -1, 2:4].norm(dim=-1) * VELOCITY_SCALE_MPS          # [B] m/s now
            scale = (0.5 + 0.5 * (speed0 - 1.4) / (11.0 - 1.4)).clamp(0.5, 1.0)     # [B]
            nan = torch.full((B,), float("nan"), device=device)
            for h_s, e, lat_t, lon_t in OFFICIAL_HORIZONS:
                pts = steps2[steps2 <= e]
                pv = gv[:, pts]                                                     # [B, P]
                n_pts = pv.sum(-1)
                ade = (dist[:, :, pts] * pv.unsqueeze(1)).sum(-1) / n_pts.clamp(min=1).unsqueeze(1)   # [B, k]
                ve = gv[:, e] > 0
                hvec = _gt_heading(gt, gv, e)                                       # [B, 2] longitudinal axis
                nvec = torch.stack([-hvec[:, 1], hvec[:, 0]], dim=-1)               # [B, 2] lateral axis
                e_err = err[:, :, e]                                                # [B, k, 2]
                lon = (e_err * hvec.unsqueeze(1)).sum(-1).abs()
                lat = (e_err * nvec.unsqueeze(1)).sum(-1).abs()
                match = (lat <= lat_t * scale[:, None]) & (lon <= lon_t * scale[:, None])   # [B, k]
                acc[f"minADE@{h_s}s"].append(torch.where(n_pts > 0, ade.min(1).values, nan).cpu())
                acc[f"minFDE@{h_s}s"].append(torch.where(ve, dist[:, :, e].min(1).values, nan).cpu())
                acc[f"MR@{h_s}s"].append(torch.where(ve, (~match.any(1)).float(), nan).cpu())
            oh = feats[:, 0, -1, 6:9]
            types.append(torch.where(oh.sum(-1) > 0, oh.argmax(-1), torch.full_like(oh[:, 0], -1).long()).cpu())
    model.train()

    per = {k: torch.cat(v).numpy() for k, v in acc.items()}
    types = torch.cat(types).numpy()
    res = {k: float(np.nanmean(v)) for k, v in per.items()}
    for n in ("minADE", "minFDE", "MR"):
        res[f"{n}_avg"] = float(np.mean([res[f"{n}@{h}s"] for h, *_ in OFFICIAL_HORIZONS]))
    res["per_type"] = {}
    for t, tn in enumerate(TYPE_SLOT_NAMES):
        m = types == t
        if m.any():
            res["per_type"][tn] = {"n": int(m.sum()), **{
                f"{n}_avg": float(np.mean([np.nanmean(per[f"{n}@{h}s"][m]) for h, *_ in OFFICIAL_HORIZONS]))
                for n in ("minADE", "minFDE", "MR")}}
    return res


def evaluate(model, scenes, name):
    """Prints the full report for one split: top-k metrics, calibration, off-road, per agent type and
    Waymo-style metrics. Returns (metrics, official_metrics)."""
    m = compute_metrics(model, scenes, return_per_sample=True)
    print(f"{name}: {len(scenes)} samples | minADE {m['minADE']:.2f} m | minFDE {m['minFDE']:.2f} m "
          f"| miss rate {m['miss_rate']:.1%} ({MISS_THRESHOLD_M} m at 8s) | top-{EVAL_TOP_K} of {NUM_MODES} modes")
    print(f"    calibration: mean top-1 confidence {m['p_top1']:.2f}, "
          f"top-1 mode is the best mode {m['top1_is_best']:.1%}")
    print(f"    off-road (vehicles, >{OFFROAD_THRESH_M:.0f} m from lane centers): top-{EVAL_TOP_K} modes "
          f"{m['offroad_rate']:.1%}, top-1 mode {m['offroad_top1']:.1%}, ground truth baseline {m['gt_offroad_rate']:.1%}")
    types = np.array([target_type(s) for s in scenes])
    ps = m["per_sample"]
    for t, tn in enumerate(TYPE_SLOT_NAMES):
        mask = types == t
        if mask.sum():
            print(f"    {tn:<10} n={mask.sum():>6}  minADE {ps['minADE'][mask].mean():.2f}  "
                  f"minFDE {ps['minFDE'][mask].mean():.2f}  MR {ps['miss'][mask].mean():.1%}")

    o = compute_official_metrics(model, scenes)
    print("    Waymo-style (2 Hz points, speed-scaled lat/lon miss rate; heading from GT direction of travel):")
    for h, *_ in OFFICIAL_HORIZONS:
        print(f"      {h} s   minADE {o[f'minADE@{h}s']:.2f}  minFDE {o[f'minFDE@{h}s']:.2f}  MR {o[f'MR@{h}s']:.1%}")
    print(f"      avg   minADE {o['minADE_avg']:.2f}  minFDE {o['minFDE_avg']:.2f}  MR {o['MR_avg']:.1%}")
    for tn, d in o["per_type"].items():
        print(f"      {tn:<10} n={d['n']:>6}  avg minADE {d['minADE_avg']:.2f}  "
              f"minFDE {d['minFDE_avg']:.2f}  MR {d['MR_avg']:.1%}")
    return m, o


def ranking_gap_table(model, scenes, batch_size=512):
    """Top-k vs perfect ranking (best of all modes) per agent type: how much accuracy the ranking loses."""
    device = _device(model)
    model.eval()
    acc = {k: [] for k in ["type", "ade6", "fde6", "ade32", "fde32"]}
    with torch.no_grad():
        for i in range(0, len(scenes), batch_size):
            b = scenes[i:i + batch_size]
            f, v, fx, fv, rx, rt, rv, h = [t.to(device) for t in collate_scenes(b)]
            pt, pc, _ = model(f, v, rx, rt, rv, h)
            pt, pc = pt.float(), pc.float()
            B = len(b)
            rows = torch.arange(B, device=device)
            dist = (pt - fx[:, None]).norm(dim=-1)                                   # [B, M, T]
            w = fv[:, None]
            ade = (dist * w).sum(-1) / w.sum(-1).clamp(min=1)                         # [B, M]
            last = (fv * torch.arange(fv.shape[1], device=device)).max(1).values.long()
            fde = dist[rows, :, last]                                                 # [B, M] at last valid step
            sel, _ = select_modes(pt, pc, EVAL_TOP_K)
            acc["ade6"].append(torch.gather(ade, 1, sel).min(1).values.cpu())
            acc["fde6"].append(torch.gather(fde, 1, sel).min(1).values.cpu())
            acc["ade32"].append(ade.min(1).values.cpu())
            acc["fde32"].append(fde.min(1).values.cpu())
            acc["type"].append(torch.tensor([target_type(s) for s in b]))
    r = {k: torch.cat(v) for k, v in acc.items()}

    def _stats(m):
        return (int(m.sum()), r["ade6"][m].mean().item(), r["fde6"][m].mean().item(),
                (r["fde6"][m] > MISS_THRESHOLD_M).float().mean().item(), r["ade32"][m].mean().item(),
                r["fde32"][m].mean().item(), (r["fde32"][m] > MISS_THRESHOLD_M).float().mean().item())

    all_ = torch.ones(len(r["type"]), dtype=torch.bool)
    rows_ = [("all", _stats(all_))] + [(n, _stats(r["type"] == t)) for t, n in enumerate(TYPE_SLOT_NAMES)]

    print(f"{'':<19}{f'top-{EVAL_TOP_K} (model ranking)':<29}best of {NUM_MODES} (perfect ranking)")
    print(f"{'type':<10}{'n':>7}  " + f"{'minADE':>8}{'minFDE':>8}{'MR':>8}     " * 2)
    for name, (n, a6, f6, m6, a32, f32, m32) in rows_:
        print(f"{name:<10}{n:>7}  {a6:>8.2f}{f6:>8.2f}{m6:>8.1%}     {a32:>8.2f}{f32:>8.2f}{m32:>8.1%}")
    model.train()
    return r
