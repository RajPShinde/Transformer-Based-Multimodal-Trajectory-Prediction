"""Anchor trajectories. Every ground-truth future gets a maneuver group from its geometry (stationary/reverse,
straight, left, right, U-turn); anchors are fit by k-means within each group. The grouped bank is the one used
for training. The base (ungrouped) shape / position anchors are kept for diagnostics and the other loss modes."""

import math

import numpy as np
import torch
from sklearn.cluster import KMeans

from config import (ANCHOR_FIT_N, ANCHOR_MAX_DIST_M, ANCHOR_SPACE, GROUP_KMEANS_STANDARDIZE, GROUP_NAMES,
                    GROUP_SLOTS, N_STRAIGHT_ANCHORS, NUM_MODES, STATIONARY_LEN_M, STRAIGHT_MAX_DEG,
                    STRAIGHT_MAX_LAT_M, TURN_DEG, UTURN_DEG, UTURN_MIN_LAT_M)
from data import target_type

assert sum(GROUP_SLOTS) == NUM_MODES and len(GROUP_SLOTS) == len(GROUP_NAMES)


def futures(scenes):
    """Stack futures -> forward-filled paths [N, T, 2], valid [N, T] bool, last valid index [N]."""
    xy = torch.stack([s["future_xy"] for s in scenes]).float().numpy()
    fv = torch.stack([s["future_valid"] for s in scenes]).numpy() > 0
    N, T, _ = xy.shape
    t = np.arange(T)
    idx = np.maximum.accumulate(np.where(fv, t, -1), axis=1)
    first = fv.argmax(axis=1)
    idx = np.where(idx < 0, first[:, None], idx)
    paths = np.take_along_axis(xy, idx[..., None], axis=1)
    last = np.where(fv, t, -1).max(axis=1)
    return paths, fv, last


def _medoid(features, members, center):
    return members[np.argmin(np.linalg.norm(features[members] - center, axis=1))]


# ---- Base anchors ----

def fit_anchor_paths(scenes, num_modes=NUM_MODES, anchor_space=ANCHOR_SPACE):
    """Base anchor bank [num_modes, T, 2] and the k-means normalizer ("position" anchors only, else {})."""
    paths, fv, last = futures(scenes)
    N, T, _ = paths.shape
    rows = np.arange(N)
    end = paths[rows, last]
    k2, k5 = paths[rows, np.minimum(20, last)], paths[rows, np.minimum(50, last)]
    full = last == T - 1
    inlier = full & (np.linalg.norm(end, axis=1) <= ANCHOR_MAX_DIST_M)
    print(f"futures valid for the full 8 s: {full.mean():.1%} of {N} (only these are used to fit)")

    if anchor_space == "position":
        shapes = np.concatenate([k2, k5, end], axis=1)
        lo, hi = shapes[inlier].min(0), shapes[inlier].max(0)
        rng = np.clip(hi - lo, 1e-3, None)
        feats = (shapes - lo) / rng
        km = KMeans(n_clusters=num_modes, n_init="auto", random_state=0).fit(feats[inlier])
        labels = km.predict(feats)
        anchors = np.zeros((num_modes, T, 2), np.float32)
        for m in range(num_modes):
            mem = np.where((labels == m) & inlier)[0]
            if len(mem):
                anchors[m] = paths[_medoid(feats, mem, km.cluster_centers_[m])]
        kf = {"centers": torch.tensor(km.cluster_centers_, dtype=torch.float32),
              "min": torch.tensor(lo, dtype=torch.float32), "range": torch.tensor(rng, dtype=torch.float32)}
        print(f"position anchors: k-means on {inlier.sum()} paths")
        return torch.from_numpy(anchors), kf

    # --- shape (intent) anchors ---
    steps = np.linalg.norm(np.diff(np.concatenate([np.zeros((N, 1, 2), np.float32), paths], axis=1), axis=1), axis=-1)
    L = steps.sum(axis=1)                                                     # path length over the 8 s
    tail = end - paths[rows, np.maximum(last - 10, 0)]                        # last ~1 s of motion
    tail = np.where((np.linalg.norm(tail, axis=1) > 0.5)[:, None], tail, end)
    theta = np.arctan2(tail[:, 1], tail[:, 0])                                # final heading
    max_lat = np.abs(paths[..., 1]).max(axis=1)

    stationary = inlier & (L < STATIONARY_LEN_M)
    moving = inlier & ~stationary
    straight = moving & (np.abs(np.degrees(theta)) < STRAIGHT_MAX_DEG) & (max_lat < STRAIGHT_MAX_LAT_M)
    other = moving & ~straight

    anchors = [np.zeros((T, 2), np.float32)]                                  # anchor 0: stationary
    s_idx = np.where(straight)[0]
    edges = np.quantile(L[s_idx], np.linspace(0, 1, N_STRAIGHT_ANCHORS + 1))
    for b in range(N_STRAIGHT_ANCHORS):
        lo_e, hi_e = edges[b], edges[b + 1]
        mem = s_idx[(L[s_idx] >= lo_e) & ((L[s_idx] <= hi_e) if b == N_STRAIGHT_ANCHORS - 1 else (L[s_idx] < hi_e))]
        med = np.median(L[mem])
        anchors.append(paths[mem[np.argmin(np.abs(L[mem] - med) + max_lat[mem])]])   # near-median length, straightest

    k_shape = num_modes - 1 - N_STRAIGHT_ANCHORS
    o_idx = np.where(other)[0]
    Lo = np.maximum(L[o_idx], 1e-3)[:, None]
    feats = np.concatenate([k2[o_idx] / Lo, k5[o_idx] / Lo, end[o_idx] / Lo,
                            np.cos(theta[o_idx])[:, None], np.sin(theta[o_idx])[:, None]], axis=1)
    km = KMeans(n_clusters=k_shape, n_init="auto", random_state=0).fit(feats)
    shape_anchors = []
    for m in range(k_shape):
        mem = np.where(km.labels_ == m)[0]
        oi = o_idx[_medoid(feats, mem, km.cluster_centers_[m])]
        shape_anchors.append((theta[oi], paths[oi]))
    shape_anchors.sort(key=lambda a: a[0])                                    # right turns ... left turns
    anchors += [p for _, p in shape_anchors]

    n_in = inlier.sum()
    print(f"shape anchors fit on {n_in} full-length paths: stationary {stationary.sum() / n_in:.1%}, "
          f"straight {straight.sum() / n_in:.1%} ({N_STRAIGHT_ANCHORS} length bins, edges "
          f"{', '.join(f'{e:.0f}' for e in edges)} m), other {other.sum() / n_in:.1%} ({k_shape} shape clusters)")
    return torch.from_numpy(np.stack(anchors).astype(np.float32)), {}


def describe_anchor_paths(anchor_paths):
    """Endpoint, path length and final heading per anchor."""
    for m in range(len(anchor_paths)):
        p = anchor_paths[m].numpy()
        seg = p[-1] - p[-11]
        L_m = np.linalg.norm(np.diff(np.concatenate([np.zeros((1, 2)), p]), axis=0), axis=1).sum()
        print(f"anchor {m:2d}: endpoint ({p[-1, 0]:6.1f}, {p[-1, 1]:6.1f}) m, path length {L_m:5.1f} m, "
              f"final heading {math.degrees(math.atan2(seg[1], seg[0])) if np.linalg.norm(seg) > 0.1 else 0.0:7.1f} deg")


def nearest_anchor_share(anchor_paths, named_splits, device, n=50_000):
    """Share of futures whose nearest anchor (by ADE) is each anchor; dead anchors show ~0%."""
    anc = anchor_paths.to(device)
    for name, split in named_splits:
        split = split[:n]
        near = []
        for i in range(0, len(split), 4096):
            chunk = split[i:i + 4096]
            fx = torch.stack([s["future_xy"] for s in chunk]).float().to(device)
            fvm = torch.stack([s["future_valid"] for s in chunk]).float().to(device)
            d = (fx[:, None] - anc[None]).norm(dim=-1)                                   # [B, M, T]
            ade = (d * fvm[:, None]).sum(-1) / fvm.sum(-1, keepdim=True).clamp(min=1)
            near.append(ade.argmin(1).cpu())
        share = np.bincount(torch.cat(near).numpy(), minlength=len(anchor_paths)) / len(split)
        print(f"{name} nearest-anchor share: {np.round(share, 3).tolist()}  (anchors under 0.2%: {(share < 0.002).sum()})")


# ---- Maneuver groups ----

def maneuver_features(paths, last):
    """paths [N, T, 2] forward-filled, last [N] last valid index -> theta (rad), L (m), r (m), max |y| (m)."""
    N = paths.shape[0]
    rows = np.arange(N)
    end = paths[rows, last]
    steps = np.linalg.norm(np.diff(np.concatenate([np.zeros((N, 1, 2), np.float32), paths], axis=1), axis=1), axis=-1)
    L = steps.sum(axis=1)
    tail = end - paths[rows, np.maximum(last - 10, 0)]
    tail = np.where((np.linalg.norm(tail, axis=1) > 0.5)[:, None], tail, end)
    theta = np.arctan2(tail[:, 1], tail[:, 0])
    small = np.abs(theta) < 1e-3
    d_arc = np.where(small, L * theta / 2, L * (1 - np.cos(theta)) / np.where(small, 1.0, theta))
    r = end[:, 1] - d_arc
    max_lat = np.abs(paths[..., 1]).max(axis=1)
    return theta, L, r, max_lat


def assign_groups(theta, L, r, max_lat):
    deg = np.degrees(theta)
    g = np.full(len(theta), 1, dtype=np.int64)                                 # straight
    g[deg > TURN_DEG] = 2
    g[deg < -TURN_DEG] = 3
    g[np.abs(deg) > UTURN_DEG] = 4
    g[(np.abs(deg) > UTURN_DEG) & (max_lat < UTURN_MIN_LAT_M)] = 0             # backing up, not a U-turn
    g[L < STATIONARY_LEN_M] = 0
    return g


def describe_groups(gt_groups, full, types):
    """Share of each maneuver group: all futures, full-length ones, and per agent type."""
    print(f"GT futures: {len(gt_groups)} (first ANCHOR_FIT_N of the seeded train shuffle); "
          f"truncated (< 8 s) {(~full).mean():.1%}")
    print(f"{'group':<20}{'all':>8}{'full 8 s':>10}{'vehicle':>10}{'ped':>8}{'cyclist':>9}")
    for k, name in enumerate(GROUP_NAMES):
        row = [(gt_groups == k).mean(), (gt_groups[full] == k).mean()]
        row += [(gt_groups[types == t] == k).mean() for t in range(3)]
        print(f"{name:<20}{row[0]:>8.1%}{row[1]:>10.1%}{row[2]:>10.1%}{row[3]:>8.1%}{row[4]:>9.1%}")


# ---- Grouped anchors (the training bank) ----

def fit_grouped_anchors(paths, last, groups, theta, L, inlier):
    """Per-group k-means on 2 / 5 / 8 s positions -> anchors [M, T, 2], owning group per slot [M]."""
    T = paths.shape[1]
    rows = np.arange(len(paths))
    k2, k5 = paths[rows, np.minimum(20, last)], paths[rows, np.minimum(50, last)]
    end = paths[rows, last]
    feats = np.concatenate([k2, k5, end], axis=1)
    anchors, owner = [], []
    for g, n in enumerate(GROUP_SLOTS):
        idx = np.where(inlier & (groups == g))[0]
        if g == 0:                                     # stationary/reverse: zero path + reversing medoid
            group_paths = [np.zeros((T, 2), np.float32)]
            rev = idx[L[idx] >= STATIONARY_LEN_M]
            if len(rev) and n > 1:
                c = feats[rev].mean(0)
                group_paths.append(paths[rev[np.argmin(np.linalg.norm(feats[rev] - c, axis=1))]])
            while len(group_paths) < n:
                group_paths.append(np.zeros((T, 2), np.float32))
        else:
            f = feats[idx]
            if GROUP_KMEANS_STANDARDIZE:
                f = (f - f.mean(0)) / (f.std(0) + 1e-6)
            km = KMeans(n_clusters=n, n_init="auto", random_state=0).fit(f)
            group_paths = []
            for m in range(n):
                loc = np.where(km.labels_ == m)[0]
                med = loc[np.argmin(np.linalg.norm(f[loc] - km.cluster_centers_[m], axis=1))]
                group_paths.append(paths[idx[med]])
            # order slots inside a group by final heading, then length
            def _key(p):
                seg = p[-1] - p[-11]
                return (round(math.degrees(math.atan2(seg[1], seg[0])) / 10), np.linalg.norm(p[-1]))
            group_paths.sort(key=_key)
        anchors += group_paths
        owner += [g] * n
        print(f"  {GROUP_NAMES[g]:<20} {n:>2} slots, fit on {len(idx):>6} full-length paths")
    return np.stack(anchors).astype(np.float32), np.array(owner)


def assign_slots(paths, last, groups, anchors, anchor_group, grouped=True):
    """Winner slot per sample: (group, then) nearest anchor point at the GT's last valid step."""
    rows = np.arange(len(paths))
    end = paths[rows, last]                                        # [N, 2]
    anc_at = anchors[:, last].transpose(1, 0, 2)                   # [N, M, 2]
    d = np.linalg.norm(anc_at - end[:, None], axis=-1)             # [N, M]
    if grouped:
        d = np.where(anchor_group[None] == groups[:, None], d, np.inf)
    w = d.argmin(1)
    return w, d[rows, w]


def describe_grouped_slots(paths, last, groups, anchors, anchor_group):
    """Assigned-slot preview: distance to the assigned anchor, how often the group rule overrides the nearest
    anchor, and each slot's geometry and winner share. Returns the assigned slots."""
    w, wd = assign_slots(paths, last, groups, anchors, anchor_group, grouped=True)
    wu, wud = assign_slots(paths, last, groups, anchors, anchor_group, grouped=False)
    share = np.bincount(w, minlength=len(anchors)) / len(w)
    print(f"\nGrouped winner: mean endpoint distance to assigned anchor {wd.mean():.2f} m "
          f"(median {np.median(wd):.2f}); unconstrained nearest anchor {wud.mean():.2f} m "
          f"(median {np.median(wud):.2f}); group rule overrides the nearest anchor in {(w != wu).mean():.1%}")
    print(f"{'slot':>4}  {'group':<20}{'endpoint (m)':>18}{'length':>9}{'heading':>9}{'winner share':>14}")
    for m in range(len(anchors)):
        p = anchors[m]
        seg = p[-1] - p[-11]
        hd = math.degrees(math.atan2(seg[1], seg[0])) if np.linalg.norm(seg) > 0.1 else 0.0
        Lm = np.linalg.norm(np.diff(np.concatenate([np.zeros((1, 2)), p]), axis=0), axis=1).sum()
        flag = "  <- under 0.2%" if share[m] < 0.002 else ""
        print(f"{m:>4}  {GROUP_NAMES[anchor_group[m]]:<20}({p[-1, 0]:7.1f},{p[-1, 1]:7.1f}){Lm:>8.1f}m{hd:>8.1f}°"
              f"{share[m]:>13.2%}{flag}")
    return w


def build_anchors(train_scenes, val_scenes, device, plot=True):
    """Fits the base anchors and the grouped training bank on the first ANCHOR_FIT_N seeded-shuffled train
    samples, printing diagnostics (and plotting them if plot=True). Returns a dict with:
      anchor_paths [M, T, 2] base bank, anchor_kf (position anchors only),
      grouped_anchor_paths [M, T, 2] and anchor_group [M] (the bank used for training),
      fit_set and assigned (the numpy-assigned slot per fit_set future, for the GPU parity check)."""
    import visualize

    fit_set = train_scenes[:ANCHOR_FIT_N]   # train_scenes was shuffled with SEED in load_splits
    anchor_paths, anchor_kf = fit_anchor_paths(fit_set)
    describe_anchor_paths(anchor_paths)
    nearest_anchor_share(anchor_paths, (("train", train_scenes), ("val", val_scenes)), device)
    if plot:
        visualize.plot_anchor_bank(anchor_paths)
    for m in range(len(anchor_paths)):
        path = anchor_paths[m].numpy()
        print(f"Anchor {m}: Max |X| = {np.abs(path[:, 0]).max():.1f}m, Max |Y| = {np.abs(path[:, 1]).max():.1f}m, "
              f"Endpoint Distance = {np.linalg.norm(path[-1]):.1f}m")

    # Maneuver groups of the ground truth and of the base anchors
    paths, _, last = futures(fit_set)
    theta, L, r, max_lat = maneuver_features(paths, last)
    gt_groups = assign_groups(theta, L, r, max_lat)
    full = last == paths.shape[1] - 1
    describe_groups(gt_groups, full, np.array([target_type(s) for s in fit_set]))
    if plot:
        visualize.plot_group_histograms(theta, L, r)
    ap = anchor_paths.numpy()
    anchor_groups = assign_groups(*maneuver_features(ap, np.full(len(ap), ap.shape[1] - 1)))
    print("\nBase anchors per group (vs. share of GT futures):")
    for k, name in enumerate(GROUP_NAMES):
        ids = np.where(anchor_groups == k)[0].tolist()
        print(f"  {name:<20} {len(ids):>2} anchors  ({(gt_groups == k).mean():5.1%} of GT)  {ids}")
    if plot:
        visualize.plot_anchors_by_group(ap, anchor_groups, title="Base anchors by maneuver group",
                                        legend=[f"{GROUP_NAMES[k]} ({(anchor_groups == k).sum()})"
                                                for k in range(len(GROUP_NAMES))])

    # Grouped anchors: the bank the model trains on
    inlier = full & (np.linalg.norm(paths[np.arange(len(paths)), last], axis=1) <= ANCHOR_MAX_DIST_M)
    print("Fitting grouped anchors:")
    ga, gown = fit_grouped_anchors(paths, last, gt_groups, theta, L, inlier)
    assigned = describe_grouped_slots(paths, last, gt_groups, ga, gown)
    if plot:
        visualize.plot_anchors_by_group(ga, gown, title="Grouped anchors",
                                        legend=[f"{GROUP_NAMES[g]} (slots {sum(GROUP_SLOTS[:g])}-"
                                                f"{sum(GROUP_SLOTS[:g + 1]) - 1})" for g in range(len(GROUP_NAMES))])
    return {"anchor_paths": anchor_paths, "anchor_kf": anchor_kf,
            "grouped_anchor_paths": torch.from_numpy(ga), "anchor_group": torch.from_numpy(gown),
            "fit_set": fit_set, "assigned": assigned}
