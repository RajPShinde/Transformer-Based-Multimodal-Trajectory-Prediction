"""Plots: road graph, raw scenes, anchor banks and maneuver groups, and the three-panel prediction plots
(top-1, scored top-6, all modes) on held-out test scenes."""

import math
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from config import (ANCHOR_SPACE, EVAL_TOP_K, GROUP_COLORS, GROUP_NAMES, LANE_CHANGE_R_M, MAX_PLAUSIBLE_COORD_M, N_VIZ_RANDOM,
                    N_VIZ_SAMPLES, N_VIZ_WORST, PLOT_SHARD_IDS, POS_SCALE_M, STATIONARY_LEN_M, TL_BUCKET_OFFSET,
                    TURN_DEG, UTURN_DEG, VIEW_MARGIN_M, VIEW_MIN_HALF_M, VIZ_MAX_ROAD_POINTS, VIZ_ROAD_RADIUS_M)
from data import (TYPE_NAMES, build_model_inputs, collate_scenes, extract_road_graph, extract_scene_agents,
                  load_dataset, to_agent_frame)

TYPE_COLORS = {0: "gray", 1: "steelblue", 2: "darkorange", 3: "seagreen", 4: "gray"}

# Plot styles per road bucket and light state
ROAD_BUCKET_STYLE = {
    1: dict(color="lightgray", kind="line", lw=1.0, label="lane center"),
    2: dict(color="khaki", kind="line", lw=1.0, label="road line"),
    3: dict(color="dimgray", kind="line", lw=1.4, label="road edge"),
    4: dict(color="red", kind="point", marker="x", size=40, label="stop sign"),
    5: dict(color="royalblue", kind="point", marker="s", size=10, label="crosswalk"),
    6: dict(color="orange", kind="point", marker="^", size=14, label="speed bump"),
    7: dict(color="silver", kind="line", lw=0.8, label="driveway"),
    0: dict(color="whitesmoke", kind="point", marker=".", size=3, label="other map"),
}
_TL_STYLE = {  # state code -> (color, marker, label)
    0: ("gray", "o", "light: unknown"),
    1: ("red", "D", "light: arrow stop"), 2: ("orange", "D", "light: arrow caution"), 3: ("limegreen", "D", "light: arrow go"),
    4: ("red", "o", "light: stop"), 5: ("orange", "o", "light: caution"), 6: ("limegreen", "o", "light: go"),
    7: ("red", "h", "light: flashing stop"), 8: ("orange", "h", "light: flashing caution"),
}
for _s, (_c, _mk, _lb) in _TL_STYLE.items():
    ROAD_BUCKET_STYLE[TL_BUCKET_OFFSET + _s] = dict(color=_c, kind="point", marker=_mk, size=70, label=_lb)

# approximate footprint (length, width) per agent type slot
CAR_SIZES = {0: (4.5, 2.0), 1: (0.7, 0.7), 2: (1.8, 0.7)}


def draw_road_graph(ax, road_xy, road_type, road_id, zorder=1):
    """Draws polylines (lane centers, road lines, road edges) as lines grouped
    by feature id, and point features and traffic lights as markers."""
    if road_xy is None or len(road_xy) == 0:
        return
    seen_labels = set()
    for bucket, style in ROAD_BUCKET_STYLE.items():
        mask = road_type == bucket
        if mask.sum() == 0:
            continue
        label = style["label"] if style["label"] not in seen_labels else None
        seen_labels.add(style["label"])

        if style["kind"] == "line":
            for feat_id in np.unique(road_id[mask]):
                seg_mask = mask & (road_id == feat_id)
                if seg_mask.sum() < 2:
                    continue
                ax.plot(
                    road_xy[seg_mask, 0], road_xy[seg_mask, 1],
                    "-", color=style["color"], linewidth=style["lw"],
                    alpha=0.9, zorder=zorder, label=label,
                )
                label = None
        else:
            is_light = bucket >= TL_BUCKET_OFFSET
            ax.scatter(
                road_xy[mask, 0], road_xy[mask, 1],
                s=style["size"], c=style["color"], marker=style["marker"],
                alpha=0.9, zorder=zorder + (3 if is_light else 0), label=label,
                edgecolors="black" if is_light else "none", linewidths=0.6,
            )


def plot_scene(scene, title="Scene"):
    """Sanity plot of one extracted sample in world coordinates."""
    fig, ax = plt.subplots(figsize=(6, 6))

    draw_road_graph(ax, scene.get("road_xy"), scene.get("road_type"), scene.get("road_id"))

    seen_labels = set()
    for i in range(scene["x"].shape[0]):
        mask = scene["valid"][i] > 0
        if mask.sum() == 0:
            continue
        xs, ys = scene["x"][i][mask], scene["y"][i][mask]
        obj_type = int(scene["type"][i])
        color = TYPE_COLORS.get(obj_type, "gray")
        type_label = TYPE_NAMES.get(obj_type, "unset")

        if i == 0:
            label = "target" if "target" not in seen_labels else None
            ax.plot(xs, ys, "o-", color="crimson", linewidth=2.5, zorder=3, label=label)
            seen_labels.add("target")
        else:
            label = f"{type_label} (context)" if type_label not in seen_labels else None
            ax.plot(xs, ys, "o-", color=color, alpha=0.6, linewidth=1, zorder=2, label=label)
            seen_labels.add(type_label)

    # target's current yaw defines the agent frame
    ox, oy = scene["origin"]
    ax.annotate("", xy=(ox + 10 * math.cos(scene["yaw0"]), oy + 10 * math.sin(scene["yaw0"])),
                xytext=(ox, oy), arrowprops=dict(arrowstyle="->", color="black", lw=2))

    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="best", fontsize=7)
    ax.set_aspect("equal")
    plt.show()


# ---- Anchors and maneuver groups ----

def plot_anchor_bank(anchor_paths, title=f"Fitted anchors ({ANCHOR_SPACE}; agent frame, forward = +x)"):
    fig, ax = plt.subplots(figsize=(6, 6))
    for m in range(len(anchor_paths)):
        ax.plot(anchor_paths[m, :, 0], anchor_paths[m, :, 1], "-", lw=1.5)
        ax.text(anchor_paths[m, -1, 0], anchor_paths[m, -1, 1], str(m), fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    ax.set_aspect("equal")
    plt.show()


def plot_group_histograms(theta, L, r):
    """Final-heading histogram (with the turn / U-turn thresholds) and lateral-residual histogram."""
    mov = L >= STATIONARY_LEN_M
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].hist(np.degrees(theta[mov]), bins=180, range=(-180, 180), color="#1f77b4")
    for x in (-UTURN_DEG, -TURN_DEG, TURN_DEG, UTURN_DEG):
        axs[0].axvline(x, color="k", ls="--", lw=0.8)
    axs[0].set_yscale("log"); axs[0].set_xlabel("final heading theta (deg)"); axs[0].set_ylabel("count (log)")
    axs[0].set_title("Final heading, moving agents")
    nt = mov & (np.abs(np.degrees(theta)) <= TURN_DEG)
    axs[1].hist(np.clip(r[nt], -10, 10), bins=200, range=(-10, 10), color="#17becf")
    for x in (-LANE_CHANGE_R_M, LANE_CHANGE_R_M):
        axs[1].axvline(x, color="k", ls="--", lw=0.8)
    axs[1].set_yscale("log"); axs[1].set_xlabel("lateral residual r = end_y - d_arc (m), clipped to +-10")
    axs[1].set_title(f"Lateral residual, |theta| <= {TURN_DEG:.0f} deg (lane changes count as straight)")
    plt.tight_layout(); plt.show()


def plot_anchors_by_group(anchors, groups, title, legend):
    """Anchors colored by maneuver group: full view and a +-30 m zoom. legend: one label per group."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    for ax, lim in zip(axs, (None, 30)):
        for m in range(len(anchors)):
            c = GROUP_COLORS[groups[m]]
            ax.plot(anchors[m, :, 0], anchors[m, :, 1], "-", color=c, lw=1.8)
            ax.plot(anchors[m, -1, 0], anchors[m, -1, 1], "o", color=c, ms=4)
            if lim is None or (abs(anchors[m, -1, 0]) < lim and abs(anchors[m, -1, 1]) < lim):
                ax.text(anchors[m, -1, 0], anchors[m, -1, 1], f" {m}", fontsize=8)
        handles = [plt.Line2D([], [], color=GROUP_COLORS[g], lw=3, label=legend[g]) for g in range(len(GROUP_NAMES))]
        ax.legend(handles=handles, fontsize=8, loc="upper left")
        if lim is not None:
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("x (m, forward)"); ax.set_ylabel("y (m, left)")
        ax.set_title(title + ("" if lim is None else f" (zoom +-{lim} m)"))
    plt.tight_layout(); plt.show()


# ---- Prediction plots ----

def is_lane_change_like(scene_inputs, angle_threshold_deg=20):
    """True if the future's heading changes by more than angle_threshold_deg (turns and sharp lane changes)."""
    fut = scene_inputs["future_xy"].numpy()
    valid = scene_inputs["future_valid"].numpy().astype(bool)
    pts = fut[valid]
    if len(pts) < 12:
        return False
    v0 = pts[5] - pts[0]
    v1 = pts[-1] - pts[-6]
    if np.linalg.norm(v0) < 1e-3 or np.linalg.norm(v1) < 1e-3:
        return False
    cos_angle = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1))) > angle_threshold_deg


def draw_faded_traj(ax, xy, color, lw=3.0, a_hi=0.85, a_lo=0.12, zorder=2):
    # thick line whose alpha fades over the horizon
    pts = xy.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n = len(segs)
    rgba = np.zeros((n, 4))
    rgba[:, :3] = to_rgb(color)
    rgba[:, 3] = np.linspace(a_hi, a_lo, n)
    ax.add_collection(LineCollection(segs, colors=rgba, linewidths=lw, zorder=zorder, capstyle="round"))


def draw_car(ax, cx, cy, theta, length, width, facecolor, edgecolor, zorder, alpha=0.9):
    # oriented footprint with a pointed front
    l, w = length / 2.0, width / 2.0
    nose = length * 0.25
    local = np.array([[l, 0.0], [l - nose, w], [-l, w], [-l, -w], [l - nose, -w]])
    c, s = math.cos(theta), math.sin(theta)
    pts = local @ np.array([[c, -s], [s, c]]).T + np.array([cx, cy])
    ax.add_patch(Polygon(pts, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=1.2, alpha=alpha, zorder=zorder))


def draw_agents(ax, agent_feats, agent_valid):
    # current-position footprints from agent-frame features (target = row 0)
    for i in range(agent_feats.shape[0]):
        vt = np.where(agent_valid[i] > 0)[0]
        if len(vt) == 0:
            continue
        t = vt[-1]
        px, py = agent_feats[i, t, 0] * POS_SCALE_M, agent_feats[i, t, 1] * POS_SCALE_M
        theta = math.atan2(agent_feats[i, t, 5], agent_feats[i, t, 4])
        oh = agent_feats[i, t, 6:9]
        ttype = int(np.argmax(oh)) if oh.sum() > 0 else -1
        length, width = CAR_SIZES.get(ttype, (2.0, 2.0))
        if i == 0:
            draw_car(ax, px, py, theta, length, width, "magenta", "darkmagenta", zorder=6)
        else:
            draw_car(ax, px, py, theta, length, width, "steelblue", "navy", zorder=5)


def plot_prediction(model, scene_inputs, viz_road_xy, viz_road_type, viz_road_id, title="Prediction", save_dir=None):
    """Three panels (top-1, scored top-k, all modes) over the wide map; saved to save_dir/<title>.png if given."""
    from metrics import select_modes

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        batch = collate_scenes([scene_inputs])
        feats, valid, _, _, road_xy, road_type, road_valid, heading = [t.to(device) for t in batch]
        pred_trajs_t, pred_confs, _ = model(feats, valid, road_xy, road_type, road_valid, heading)
        sel_idx, _ = select_modes(pred_trajs_t, pred_confs, EVAL_TOP_K)
        order = sel_idx[0].cpu().numpy()
        probs = F.softmax(pred_confs, dim=-1)[0].cpu().numpy()
        pred_trajs = pred_trajs_t[0].cpu().numpy()
    model.train()

    agent_feats = scene_inputs["agent_feats"].float().numpy()   # [A, T, F], agent frame
    agent_valid = scene_inputs["agent_valid"].float().numpy()   # [A, T]
    gt = scene_inputs["future_xy"].numpy()
    gt_valid = scene_inputs["future_valid"].numpy().astype(bool)
    past = agent_feats[0, :, :2] * POS_SCALE_M
    past_valid = agent_valid[0] > 0
    M = pred_trajs.shape[0]

    # one square window framed on ground truth, history and the scored top-6
    framed = np.concatenate([pred_trajs[order].reshape(-1, 2), gt[gt_valid], past[past_valid], np.zeros((1, 2))])
    cx = 0.5 * (framed[:, 0].min() + framed[:, 0].max())
    cy = 0.5 * (framed[:, 1].min() + framed[:, 1].max())
    half = max(0.5 * max(np.ptp(framed[:, 0]), np.ptp(framed[:, 1])) + VIEW_MARGIN_M, VIEW_MIN_HALF_M)

    # (title, mode indices, show probability legend)
    panels = [("top-1", order[:1], True),
              (f"top-{len(order)} (scored)", order, True),
              (f"all {M} modes (far ones clipped)", np.arange(M), False)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    cmap = plt.cm.viridis
    for ax, (ptitle, idxs, show_probs) in zip(axes, panels):
        draw_road_graph(ax, viz_road_xy, viz_road_type, viz_road_id)
        mode_handles = []
        for j, m in enumerate(idxs):
            color = cmap(j / max(len(idxs) - 1, 1))
            draw_faded_traj(ax, pred_trajs[m], color, lw=3.0)
            if show_probs:
                mode_handles.append(Line2D([0], [0], color=color, lw=3.0, label=f"mode {m}: p={probs[m]:.2f}"))
        # history, agent footprints, then ground truth on top
        ax.plot(past[past_valid, 0], past[past_valid, 1], "-", color="red", linewidth=2.0, zorder=3)
        draw_agents(ax, agent_feats, agent_valid)
        ax.plot(gt[gt_valid, 0], gt[gt_valid, 1], "-", color="black", linewidth=1.4, zorder=20)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(ptitle, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        if show_probs:
            ax.legend(handles=mode_handles, loc="upper right", fontsize=7, framealpha=0.85)

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="magenta",
               markeredgecolor="darkmagenta", markersize=11, label="target"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="steelblue",
               markeredgecolor="navy", markersize=10, label="other agents"),
        Line2D([0], [0], color="red", lw=2.0, label="history (1.1 s)"),
        Line2D([0], [0], color="black", lw=1.4, label="ground truth (future)"),
        Line2D([0], [0], color="0.25", lw=3.0, alpha=0.7, label="predicted modes (fade = time elapsed)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.015), fontsize=10)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    if save_dir:
        fname = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_") + ".png"
        fig.savefig(os.path.join(save_dir, fname), dpi=130, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def viz_candidates(shard_paths):
    """One sample (the first target) per scene in the given raw shards, with a wide square map crop and the
    traffic lights overlaid, in the target's agent frame -> list of (scene_inputs, xy, type, id)."""
    out = []
    for parsed_scene in load_dataset(shard_paths):
        samples = extract_scene_agents(parsed_scene, max_targets=1)
        if not samples:
            continue
        scene = samples[0]
        scene_inputs = build_model_inputs(scene)
        fv = scene_inputs["future_valid"].numpy() > 0
        if fv.sum() == 0 or np.linalg.norm(scene_inputs["future_xy"].numpy()[fv], axis=1).max() > MAX_PLAUSIBLE_COORD_M:
            continue
        wide_xy_raw, wide_type, wide_id, _ = extract_road_graph(
            parsed_scene, scene["origin"], radius=VIZ_ROAD_RADIUS_M, max_points=VIZ_MAX_ROAD_POINTS, square=True
        )
        wide_xy = to_agent_frame(wide_xy_raw, scene["origin"], scene["yaw0"])
        if len(scene["tl_xy"]):   # overlay traffic lights
            wide_xy = np.concatenate([wide_xy, to_agent_frame(scene["tl_xy"], scene["origin"], scene["yaw0"])])
            wide_type = np.concatenate([wide_type, scene["tl_type"]])
            wide_id = np.concatenate([wide_id, np.full(len(scene["tl_xy"]), -1, dtype=np.int64)])
        out.append((scene_inputs, wide_xy, wide_type, wide_id))
    return out


def plot_test_predictions(model, save_dir, shard_ids=PLOT_SHARD_IDS):
    """Prediction plots on held-out test shards: turn-like, random (fixed seed) and worst by top-6 minADE.
    Downloads the raw shards first (needs `gcloud auth login`)."""
    from metrics import compute_metrics
    from prep import download_shards, local_shard

    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving plots to {save_dir}")
    plot_paths = []
    for i in shard_ids:
        download_shards([i])
        if os.path.exists(local_shard(i)):
            plot_paths.append(local_shard(i))
        else:
            print(f"shard {i}: download failed -- run `gcloud auth login`, then try again")

    candidates = viz_candidates(plot_paths)
    turn_like = [c for c in candidates if is_lane_change_like(c[0])]
    print(f"{len(candidates)} TEST candidates, {len(turn_like)} turn-like")
    if not candidates:
        raise RuntimeError("No scenes to plot -- the shard downloads failed. Run `gcloud auth login`, then try again.")

    # Turn-like scenes
    to_show = turn_like[:N_VIZ_SAMPLES] if len(turn_like) >= N_VIZ_SAMPLES else candidates[:N_VIZ_SAMPLES]
    for idx, (scene_inputs, wide_xy, wide_type, wide_id) in enumerate(to_show):
        plot_prediction(model, scene_inputs, wide_xy, wide_type, wide_id, title=f"TEST turn-like {idx:02d}",
                        save_dir=save_dir)

    # Random scenes (fixed seed)
    for idx, (scene_inputs, wide_xy, wide_type, wide_id) in enumerate(
            random.Random(0).sample(candidates, min(N_VIZ_RANDOM, len(candidates)))):
        plot_prediction(model, scene_inputs, wide_xy, wide_type, wide_id, title=f"TEST random {idx:02d}",
                        save_dir=save_dir)

    # Worst scenes by top-6 minADE
    cand_ade = compute_metrics(model, [c[0] for c in candidates], return_per_sample=True)["per_sample"]["minADE"]
    for rank, ci in enumerate(np.argsort(-cand_ade)[:N_VIZ_WORST]):
        scene_inputs, wide_xy, wide_type, wide_id = candidates[ci]
        plot_prediction(model, scene_inputs, wide_xy, wide_type, wide_id,
                        title=f"TEST worst {rank:02d} (minADE {cand_ade[ci]:.1f} m)", save_dir=save_dir)
