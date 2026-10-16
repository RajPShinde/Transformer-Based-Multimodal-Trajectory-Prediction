"""Loss: Gaussian NLL on the winning mode + confidence cross-entropy.

The shipped setting is ASSIGNMENT_MODE = "grouped" with GROUPED_HYBRID: the ground-truth maneuver group is fixed,
the closest predicted slot inside that group gets the NLL, and the confidence target is zero outside the group and
a relative-tau softmax over in-group ADE inside it. The other assignment modes and confidence targets are the
ablations tried along the way; switch them with the flags in config.py (read at call time).

`bank` is a dict holding the anchors the loss needs: group_anchors [M, T, 2] and anchor_group [M] for "grouped",
anchor_kf for "anchor". model.load_checkpoint returns one; train.train builds one from the fitted anchors."""

import torch
import torch.nn.functional as F

import config
from config import STATIONARY_LEN_M, TURN_DEG, UTURN_DEG, UTURN_MIN_LAT_M


def gt_groups_torch(target_xy, target_valid):
    """GPU twin of anchors.maneuver_features + anchors.assign_groups -> group [B], endpoint [B, 2], last [B].
    Same forward-fill, heading from the last ~1 s, path length from the origin, same thresholds."""
    B, T, _ = target_xy.shape
    dev = target_xy.device
    valid = target_valid > 0
    t = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
    lv = torch.where(valid, t, torch.full_like(t, -1)).cummax(dim=1).values                # [B, T]
    first = valid.float().argmax(dim=1, keepdim=True)
    src = torch.where(lv < 0, first.expand(B, T), lv)
    traj = torch.gather(target_xy, 1, src.unsqueeze(-1).expand(B, T, 2))                   # forward-filled
    last = lv[:, -1].clamp(min=0)
    rows = torch.arange(B, device=dev)
    end = traj[rows, last]
    L = torch.diff(torch.cat([torch.zeros(B, 1, 2, device=dev), traj], dim=1), dim=1).norm(dim=-1).sum(1)
    tail = end - traj[rows, (last - 10).clamp(min=0)]
    tail = torch.where(tail.norm(dim=-1, keepdim=True) > 0.5, tail, end)
    deg = torch.rad2deg(torch.atan2(tail[:, 1], tail[:, 0]))
    max_lat = traj[..., 1].abs().amax(dim=1)
    g = torch.ones(B, dtype=torch.long, device=dev)                                         # straight
    g[deg > TURN_DEG] = 2
    g[deg < -TURN_DEG] = 3
    g[deg.abs() > UTURN_DEG] = 4
    g[(deg.abs() > UTURN_DEG) & (max_lat < UTURN_MIN_LAT_M)] = 0                             # backing up
    g[L < STATIONARY_LEN_M] = 0
    return g, end, last


def grouped_winner(target_xy, target_valid, anchors, anchor_group):
    """Assigned slot [B]: maneuver group from the GT, then the nearest anchor (at the GT's last valid
    step) among that group's slots. Twin of anchors.assign_slots."""
    if anchors is None or anchor_group is None:
        raise RuntimeError("group_anchors / anchor_group not set: fit grouped anchors or load a grouped checkpoint.")
    g, end, last = gt_groups_torch(target_xy.float(), target_valid)
    anc_at = anchors[:, last].permute(1, 0, 2)                                              # [B, M, 2]
    d = (anc_at - end.unsqueeze(1)).norm(dim=-1)                                            # [B, M]
    d = d.masked_fill(anchor_group.unsqueeze(0) != g.unsqueeze(1), float("inf"))
    return d.argmin(dim=1)


def anchor_labels(target_xy, target_valid, kf):
    """Ground truth -> anchor index [B], matching kmeans.predict at fit time:
    forward-fill invalid steps, take (2s, 5s, last-valid) keyframes,
    min-max normalize, nearest cluster center."""
    if not kf:
        raise RuntimeError("anchor_kf is not set (only 'position' anchors define it): "
                           "use ASSIGNMENT_MODE = 'trajectory' or fit position anchors.")
    B, T, _ = target_xy.shape
    valid = target_valid > 0
    t = torch.arange(T, device=target_xy.device).unsqueeze(0).expand(B, T)
    last_valid_le_t = torch.where(valid, t, torch.full_like(t, -1)).cummax(dim=1).values   # [B, T]
    first_valid = valid.float().argmax(dim=1, keepdim=True)                                # [B, 1]
    src = torch.where(last_valid_le_t < 0, first_valid.expand(B, T), last_valid_le_t)
    traj = torch.gather(target_xy, 1, src.unsqueeze(-1).expand(B, T, 2))                  # forward-filled

    last = last_valid_le_t[:, -1].clamp(min=0)
    rows = torch.arange(B, device=target_xy.device)
    k2 = torch.clamp(last, max=20)
    k5 = torch.clamp(last, max=50)
    feats = torch.cat([traj[rows, k2], traj[rows, k5], traj[rows, last]], dim=-1)         # [B, 6]
    norm = (feats - kf["min"]) / kf["range"]
    return torch.cdist(norm, kf["centers"]).argmin(dim=1)                                  # [B]


def relative_tau(ade_min):
    """Softmax temperature that grows with the best mode's error (m)."""
    return (config.REL_TAU_SCALE * ade_min + config.REL_TAU_BIAS).clamp(min=config.REL_TAU_BIAS, max=config.REL_TAU_MAX)


def confidence_target(ade, weights=None, mode=None):
    """ade [B, M] (no grad) -> confidence target [B, M]. `weights` is the regression assignment
    (one-hot for hard modes), used only by the "hard" target."""
    mode = config.CONF_TARGET if mode is None else mode
    if mode == "relative":
        ade_min = ade.min(dim=1, keepdim=True).values                                       # [B, 1]
        return F.softmax(-(ade - ade_min) / relative_tau(ade_min), dim=1)
    if mode == "fixed":
        return F.softmax(-ade / config.CONF_TARGET_TEMP_M, dim=1)
    if mode == "hard":
        M = ade.shape[1]
        if weights is None:
            weights = F.one_hot(ade.argmin(dim=1), M).float()
        return (1.0 - config.LABEL_SMOOTHING) * weights + config.LABEL_SMOOTHING / M
    raise ValueError(f"Unknown CONF_TARGET: {mode}")


def gaussian_nll_loss(pred_trajs, pred_confs, pred_log_std, target_xy, target_valid, bank=None, use_learned_std=True):
    """Returns (total_loss, nll_loss, conf_loss, stats); nll / conf are detached and unweighted,
    stats holds detached per-batch diagnostics (confidence sharpness, target entropy, ranking accuracy)."""
    # pred_trajs [B, M, T, 2], pred_confs [B, M], pred_log_std [B, M, T]; target_xy [B, T, 2], target_valid [B, T]
    mode = config.ASSIGNMENT_MODE
    hybrid = mode == "grouped" and config.GROUPED_HYBRID
    bank = bank or {}
    B, M, T, _ = pred_trajs.shape
    batch_idx = torch.arange(B, device=pred_trajs.device)

    sq_dist = ((pred_trajs - target_xy.unsqueeze(1)) ** 2).sum(-1)   # [B, M, T]
    vmask = target_valid.unsqueeze(1)                                  # [B, 1, T]
    denom = target_valid.sum(-1).clamp(min=1.0).unsqueeze(1)           # [B, 1]

    # Isotropic 2D Gaussian NLL per mode and timestep (constant term dropped)
    log_std = pred_log_std if use_learned_std else torch.zeros_like(pred_log_std)  # sigma=1 during warmup
    nll_per_t = 2.0 * log_std + sq_dist * torch.exp(-2.0 * log_std) / 2.0         # [B, M, T]
    nll_per_mode = (nll_per_t * vmask).sum(-1) / denom                             # [B, M]

    # ADE of every mode to the truth (no gradient), used for winners and soft targets
    with torch.no_grad():
        ade = (sq_dist.clamp(min=0.0).sqrt() * vmask).sum(-1) / denom             # [B, M]

    winner = None
    assigned = None          # grouped only: the anchor-assigned slot (from the GT)
    same_group = None        # grouped only: [B, M] True for slots in the GT's group
    if mode == "soft":
        # soft responsibilities are fixed targets (no gradient through the weights)
        weights = F.softmax(-ade / config.SOFT_ASSIGN_TEMP_M, dim=1)              # [B, M]
    else:
        if mode == "grouped":
            anchor_group = bank.get("anchor_group")
            with torch.no_grad():
                assigned = grouped_winner(target_xy, target_valid, bank.get("group_anchors"), anchor_group)  # [B]
                same_group = anchor_group.unsqueeze(0) == anchor_group[assigned].unsqueeze(1)  # [B, M]
                if hybrid:
                    # closest predicted slot within the GT's maneuver group
                    winner = ade.masked_fill(~same_group, float("inf")).argmin(dim=1)              # [B]
                else:
                    winner = assigned
        elif mode == "trajectory":
            winner = ade.argmin(dim=1)                                             # [B]
        elif mode == "anchor":
            winner = anchor_labels(target_xy, target_valid, bank.get("anchor_kf"))  # [B]
        elif mode == "endpoint":
            # last valid step (validity is not always a contiguous prefix)
            time_idx = torch.arange(T, device=target_valid.device)
            last_idx = (target_valid * time_idx.unsqueeze(0)).max(dim=1).values.long()  # [B]
            winner = sq_dist[batch_idx, :, last_idx].argmin(dim=1)                      # [B]
        else:
            raise ValueError(f"Unknown ASSIGNMENT_MODE: {mode}")
        weights = F.one_hot(winner, M).float()

    nll_loss = (weights * nll_per_mode).sum(dim=1).mean()

    with torch.no_grad():
        if hybrid:
            # hybrid target: zero outside the GT group, relative-tau softmax over ADE inside it
            ade_g = ade.masked_fill(~same_group, float("inf"))
            ade_min = ade_g.min(dim=1, keepdim=True).values
            conf_target = F.softmax(-(ade_g - ade_min) / relative_tau(ade_min), dim=1)   # zero outside the group
        else:
            conf_target = confidence_target(ade, weights)                         # [B, M]
    conf_loss = -(conf_target * F.log_softmax(pred_confs, dim=-1)).sum(dim=-1).mean()

    total_loss = config.NLL_WEIGHT * nll_loss + config.CONF_WEIGHT * conf_loss

    stats = {}
    with torch.no_grad():
        stats["p_top1"] = F.softmax(pred_confs, dim=-1).max(dim=1).values.mean()
        stats["tgt_entropy"] = -(conf_target * conf_target.clamp(min=1e-12).log()).sum(-1).mean()
        stats["tgt_win"] = conf_target.gather(1, ade.argmin(dim=1, keepdim=True)).mean()
        if mode == "grouped":
            top1 = pred_confs.argmax(dim=1)
            stats["assigned_top1"] = (top1 == assigned).float().mean()
            stats["group_top1"] = same_group.gather(1, top1.unsqueeze(1)).float().mean()
            stats["reg_is_assigned"] = (winner == assigned).float().mean()
            stats["rank_top1"] = stats["group_top1"] if hybrid else stats["assigned_top1"]

    return total_loss, nll_loss.detach(), conf_loss.detach(), stats
