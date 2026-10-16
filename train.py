"""Training: time-boxed AdamW with linear warmup + cosine LR, GPU-resident data, bf16 autocast, resumable atomic
checkpoints every CHECKPOINT_EVERY_SEC, and the best checkpoint kept by validation minADE."""

import math
import os
import random
import time

import numpy as np
import torch

import config
from config import (ANCHOR_SPACE, BATCH_SIZE, CHECKPOINT_EVERY_SEC, CONF_TARGET_TEMP_M, CUMULATIVE_OFFSETS, D_MODEL,
                    DECODER_FF_DIM, DECODER_ROUNDS, DRIVE_DIR, DROPOUT, EVAL_BATCH_SIZE, EVAL_TOP_K, FF_DIM,
                    FUTURE_STEPS, GRAD_CLIP_NORM, GROUP_KMEANS_STANDARDIZE, GROUP_NAMES, GROUP_QUERY_EMBED,
                    GROUP_SLOTS, HIST_STEPS, LABEL_SMOOTHING, LOG_EVERY_STEPS, LOG_STD_MAX, LOG_STD_MIN, MAX_AGENTS,
                    MAX_ROAD_POINTS, MAX_TRAFFIC_LIGHTS, MIN_LR, N_HEADS, N_LAYERS, NLL_WARMUP_STEPS, NMS_DIST_M,
                    NUM_MODES, NUM_ROAD_BUCKETS, ORACLE_VAL_N, PEAK_LR, POS_SCALE_M, PREP_OUT_DIR, AGENT_FEAT_DIM,
                    ROAD_FEAT_DIM, ROAD_FORWARD_OFFSET_M, ROAD_POINT_STRIDE, ROAD_RADIUS_M, SEED, STATIONARY_LEN_M,
                    TL_BUCKET_OFFSET, TRAIN_TIME_BUDGET_SEC, TURN_DEG, UTURN_DEG, UTURN_MIN_LAT_M,
                    VELOCITY_SCALE_MPS, WARMUP_STEPS, WEIGHT_DECAY)
from data import collate_scenes
from loss import gaussian_nll_loss, grouped_winner
from metrics import compute_metrics
from model import TrajectoryTransformer
from prep import split_shard_counts

CHECKPOINT_NAME = "trajectory_transformer_checkpoint.pt"   # periodic, resumable
BEST_CHECKPOINT_NAME = "trajectory_transformer_best.pt"    # best val minADE


def lr_schedule(step, elapsed_sec, budget_sec=TRAIN_TIME_BUDGET_SEC):
    """Linear warmup over WARMUP_STEPS, times a cosine decay from PEAK_LR to MIN_LR over the time budget."""
    warm = min(1.0, (step + 1) / WARMUP_STEPS)
    progress = min(1.0, elapsed_sec / budget_sec)
    return warm * (MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress)))


# ---- data on the GPU ----
_BANK_KEYS = ["agent_feats", "agent_valid", "future_xy", "future_valid", "road_xy", "road_type", "road_valid"]


def make_gpu_bank(scenes, device, chunk=100_000):
    bank = {}
    for k in _BANK_KEYS:
        parts = [torch.stack([s[k] for s in scenes[i:i + chunk]]).to(device) for i in range(0, len(scenes), chunk)]
        bank[k] = torch.cat(parts)
        del parts
    return bank


def bank_batch(bank, idx):
    """Same 8-tuple as collate_scenes, sliced from a GPU bank and cast to the model's dtypes."""
    heading = torch.tensor([1.0, 0.0], device=idx.device)
    return (bank["agent_feats"][idx].float(), bank["agent_valid"][idx].float(),
            bank["future_xy"][idx].float(), bank["future_valid"][idx].float(),
            bank["road_xy"][idx].float(), bank["road_type"][idx].long(),
            bank["road_valid"][idx].float(), heading.expand(len(idx), 2))


def compute_val_loss(model, bank, loss_bank, use_learned_std=True, batch_size=EVAL_BATCH_SIZE, use_bf16=False):
    """Training loss on held-out samples (reported only; selection uses val minADE).
    Returns (loss, rank_top1); rank_top1 is nan unless ASSIGNMENT_MODE == "grouped"."""
    device = bank["agent_feats"].device
    model.eval()
    total, nb = torch.zeros((), device=device), 0
    atop1 = torch.zeros((), device=device)
    n = bank["agent_feats"].shape[0]
    with torch.no_grad():
        for i in range(0, n - batch_size + 1, batch_size):
            idx = torch.arange(i, i + batch_size, device=device)
            f, v, fx, fv, rx, rt, rv, h = bank_batch(bank, idx)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                pt, pc, pl = model(f, v, rx, rt, rv, h)
            loss, _, _, stats = gaussian_nll_loss(pt, pc, pl, fx, fv, loss_bank, use_learned_std=use_learned_std)
            total += loss
            atop1 += stats.get("rank_top1", torch.full((), float("nan"), device=device))
            nb += 1
    model.train()
    return float(total) / max(nb, 1), float(atop1) / max(nb, 1)


def oracle_minade(model, bank, n=ORACLE_VAL_N, batch_size=EVAL_BATCH_SIZE, use_bf16=False):
    """Best-of-ALL-modes minADE on the first n samples: regression quality, independent of ranking."""
    device = bank["agent_feats"].device
    model.eval()
    total = torch.zeros((), device=device)
    n = min(n, bank["agent_feats"].shape[0])
    with torch.no_grad():
        for i in range(0, n, batch_size):
            idx = torch.arange(i, min(i + batch_size, n), device=device)
            f, v, fx, fv, rx, rt, rv, h = bank_batch(bank, idx)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                pt, _, _ = model(f, v, rx, rt, rv, h)
            w = fv[:, None]
            ade = ((pt - fx[:, None]).norm(dim=-1) * w).sum(-1) / w.sum(-1).clamp(min=1)
            total += ade.min(1).values.sum()
    model.train()
    return float(total) / n


def smoke_test(train_scenes, anchors, device, steps=300, batch_size=128):
    """(a) the GPU winner rule matches the numpy rule on the anchor fit set; (b) a short training run with the
    current loss stays finite."""
    loss_bank = {"group_anchors": anchors["grouped_anchor_paths"].to(device),
                 "anchor_group": anchors["anchor_group"].to(device), "anchor_kf": {}}

    # (a) parity
    fit_set, w = anchors["fit_set"], anchors["assigned"]
    tw = []
    with torch.no_grad():
        for i in range(0, len(fit_set), 8192):
            ch = fit_set[i:i + 8192]
            fx = torch.stack([s["future_xy"] for s in ch]).float().to(device)
            fv = torch.stack([s["future_valid"] for s in ch]).float().to(device)
            tw.append(grouped_winner(fx, fv, loss_bank["group_anchors"], loss_bank["anchor_group"]).cpu())
    tw = torch.cat(tw).numpy()
    print(f"(a) GPU vs numpy assigned slot agreement on {len(tw)} futures: {(tw == w).mean():.4%} "
          f"({(tw != w).sum()} differ)")

    # (b) smoke training
    print(f"(b) ASSIGNMENT_MODE={config.ASSIGNMENT_MODE} GROUPED_HYBRID={config.GROUPED_HYBRID} "
          f"CONF_TARGET={config.CONF_TARGET}")
    torch.manual_seed(0)
    sm = TrajectoryTransformer(anchor_paths=anchors["grouped_anchor_paths"], anchor_group=anchors["anchor_group"],
                               num_groups=len(GROUP_NAMES) if GROUP_QUERY_EMBED else 0).to(device)
    opt = torch.optim.AdamW(sm.parameters(), lr=3e-4, weight_decay=1e-4)
    sm.train()
    log = []
    t = time.time()
    for step in range(steps):
        lr = 3e-4 * min(1.0, (step + 1) / 100)
        for g_ in opt.param_groups:
            g_["lr"] = lr
        batch = train_scenes[(step * batch_size) % 200_000:(step * batch_size) % 200_000 + batch_size]
        f, v, fx, fv, rx, rt, rv, h = [x.to(device) for x in collate_scenes(batch)]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pt, pc, pl = sm(f, v, rx, rt, rv, h)
        loss, nll, conf, stats = gaussian_nll_loss(pt, pc, pl, fx, fv, loss_bank, use_learned_std=False)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sm.parameters(), 1.0)
        opt.step()
        log.append([float(loss.detach()), float(nll), float(conf), float(stats["group_top1"]),
                    float(stats["assigned_top1"]), float(stats["reg_is_assigned"]), float(stats["tgt_entropy"])])
        if (step + 1) % 50 == 0:
            a = np.array(log[-50:]).mean(0)
            print(f"  step {step + 1}: loss {a[0]:.3f} nll {a[1]:.3f} conf {a[2]:.3f} (target H {a[6]:.2f}) "
                  f"gtop1 {a[3]:.1%} atop1 {a[4]:.1%} reg=asg {a[5]:.1%}", flush=True)
    assert all(np.isfinite(np.array(log)).ravel()), "non-finite values in the smoke log"
    print(f"(b) smoke OK: {steps} steps in {time.time() - t:.0f} s | shapes: trajs {tuple(pt.shape)}, "
          f"confs {tuple(pc.shape)} | group embedding: {sm.num_groups} groups")
    del sm, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _run_config(run_name, use_bf16, n_train, n_val, n_test, anchors):
    """Everything needed to reproduce the run, saved inside each checkpoint."""
    hybrid = config.ASSIGNMENT_MODE == "grouped" and config.GROUPED_HYBRID
    n_train_shards, n_val_shards, n_test_shards = split_shard_counts()
    cfg = dict(
        run_name=run_name, seed=SEED, prepped_dir=PREP_OUT_DIR, anchor_space=ANCHOR_SPACE,
        assignment_mode=config.ASSIGNMENT_MODE, conf_target=config.CONF_TARGET, rel_tau_scale=config.REL_TAU_SCALE,
        rel_tau_bias=config.REL_TAU_BIAS, rel_tau_max=config.REL_TAU_MAX, soft_conf_target=config.CONF_TARGET != "hard",
        conf_target_temp_m=CONF_TARGET_TEMP_M, label_smoothing=LABEL_SMOOTHING,
        nll_weight=config.NLL_WEIGHT, conf_weight=config.CONF_WEIGHT,
        cumulative_offsets=CUMULATIVE_OFFSETS, EVAL_TOP_K=EVAL_TOP_K, NMS_DIST_M=NMS_DIST_M,
        batch_size=BATCH_SIZE, peak_lr=PEAK_LR, min_lr=MIN_LR, warmup_steps=WARMUP_STEPS,
        nll_warmup_steps=NLL_WARMUP_STEPS, weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP_NORM,
        lr_schedule="warmup + cosine over time budget", time_budget_sec=TRAIN_TIME_BUDGET_SEC, use_bf16=use_bf16,
        MAX_AGENTS=MAX_AGENTS, HIST_STEPS=HIST_STEPS, FUTURE_STEPS=FUTURE_STEPS,
        MAX_ROAD_POINTS=MAX_ROAD_POINTS, MAX_TRAFFIC_LIGHTS=MAX_TRAFFIC_LIGHTS, ROAD_RADIUS_M=ROAD_RADIUS_M,
        ROAD_POINT_STRIDE=ROAD_POINT_STRIDE, ROAD_FORWARD_OFFSET_M=ROAD_FORWARD_OFFSET_M,
        POS_SCALE_M=POS_SCALE_M, VELOCITY_SCALE_MPS=VELOCITY_SCALE_MPS,
        AGENT_FEAT_DIM=AGENT_FEAT_DIM, ROAD_FEAT_DIM=ROAD_FEAT_DIM, NUM_ROAD_BUCKETS=NUM_ROAD_BUCKETS,
        TL_BUCKET_OFFSET=TL_BUCKET_OFFSET, NUM_MODES=NUM_MODES, D_MODEL=D_MODEL, N_HEADS=N_HEADS,
        N_LAYERS=N_LAYERS, DECODER_ROUNDS=DECODER_ROUNDS, FF_DIM=FF_DIM, DECODER_FF_DIM=DECODER_FF_DIM,
        DROPOUT=DROPOUT, LOG_STD_MIN=LOG_STD_MIN, LOG_STD_MAX=LOG_STD_MAX, anchor_conditioned_queries=True,
        split=(f"by shard id%10: 8 = val, 9 = test, else train "
               f"({n_train_shards} / {n_val_shards} / {n_test_shards} shards)"),
        n_train=n_train, n_val=n_val, n_test=n_test,
    )
    if config.ASSIGNMENT_MODE == "grouped":
        cfg.update(anchor_bank="grouped", group_names=GROUP_NAMES, group_slots=GROUP_SLOTS,
                   group_kmeans_standardize=GROUP_KMEANS_STANDARDIZE, turn_deg=TURN_DEG, uturn_deg=UTURN_DEG,
                   uturn_min_lat_m=UTURN_MIN_LAT_M, stationary_len_m=STATIONARY_LEN_M,
                   group_query_embed=GROUP_QUERY_EMBED, anchor_group=anchors["anchor_group"].tolist(),
                   grouped_hybrid=config.GROUPED_HYBRID)
    return cfg, hybrid


def train(train_scenes, val_scenes, anchors, device, resume_run_dir=None, n_test=0):
    """Trains until TRAIN_TIME_BUDGET_SEC (total across resumes). Pass resume_run_dir to continue a run.
    anchors: the dict from anchors.build_anchors. Returns the path of the best checkpoint."""
    if resume_run_dir:
        ckpt_dir = resume_run_dir
        run_name = os.path.basename(resume_run_dir.rstrip("/"))
    else:
        run_name = time.strftime("run_%Y%m%d_%H%M")
        ckpt_dir = f"{DRIVE_DIR}/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = f"{ckpt_dir}/{CHECKPOINT_NAME}"
    best_checkpoint_path = f"{ckpt_dir}/{BEST_CHECKPOINT_NAME}"

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32 for remaining fp32 matmuls (Ampere+)
    torch.backends.cudnn.allow_tf32 = True

    t0 = time.time()
    train_bank = make_gpu_bank(train_scenes, device)
    val_bank = make_gpu_bank(val_scenes, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    n_train = len(train_scenes)
    print(f"GPU banks ready in {(time.time() - t0) / 60:.1f} min: train {n_train}, val {len(val_scenes)}" +
          (f" | GPU memory allocated {torch.cuda.memory_allocated() / 1e9:.1f} GB" if device.type == "cuda" else ""))

    run_config, hybrid = _run_config(run_name, use_bf16, n_train, len(val_scenes), n_test, anchors)
    conf_desc = {"relative": f"relative tau = {config.REL_TAU_SCALE}*ade_min + {config.REL_TAU_BIAS} "
                             f"(max {config.REL_TAU_MAX}) m",
                 "fixed": f"fixed tau = {CONF_TARGET_TEMP_M} m",
                 "hard": f"hard (label smoothing {LABEL_SMOOTHING})"}[config.CONF_TARGET]
    if hybrid:
        conf_desc = (f"grouped hybrid: GT group fixed, in-group relative tau = {config.REL_TAU_SCALE}*ade_min + "
                     f"{config.REL_TAU_BIAS} (max {config.REL_TAU_MAX}) m; regression = closest slot in the GT group")

    # ---- anchors: the grouped bank when ASSIGNMENT_MODE == "grouped", else the base bank ----
    anchor_kf = {k: v.to(device) for k, v in anchors["anchor_kf"].items()}
    model_kw = {}
    if config.ASSIGNMENT_MODE == "grouped":
        anchor_paths = anchors["grouped_anchor_paths"]   # the bank this run trains on (saved in checkpoints)
        loss_bank = {"group_anchors": anchor_paths.to(device), "anchor_group": anchors["anchor_group"].to(device),
                     "anchor_kf": anchor_kf}
        model_kw = dict(anchor_group=anchors["anchor_group"],
                        num_groups=len(GROUP_NAMES) if GROUP_QUERY_EMBED else 0)
        print(f"Grouped anchors: slots per group {dict(zip(GROUP_NAMES, GROUP_SLOTS))}, "
              f"group query embedding {'on' if GROUP_QUERY_EMBED else 'off'}")
    else:
        anchor_paths = anchors["anchor_paths"]
        loss_bank = {"group_anchors": None, "anchor_group": None, "anchor_kf": anchor_kf}

    # ---- model, optimizer, resume ----
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = TrajectoryTransformer(anchor_paths=anchor_paths, **model_kw).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    state = dict(epoch=0, batch_pos=0, global_step=0, elapsed=0.0, best_val_minade=float("inf"))

    if resume_run_dir and os.path.exists(checkpoint_path):
        ck = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ck["model_state_dict"])          # includes the run's anchor buffer
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        anchor_paths = ck["anchor_paths"]
        loss_bank["anchor_kf"] = {k: v.to(device) for k, v in ck["anchor_kf"].items()}
        if config.ASSIGNMENT_MODE == "grouped":
            loss_bank["group_anchors"] = anchor_paths.to(device)
            loss_bank["anchor_group"] = torch.as_tensor(ck["anchor_group"]).long().to(device)
        state.update(ck["train_state"])
        print(f"Resumed {run_name}: epoch {state['epoch']} + {state['batch_pos']} batches, step {state['global_step']}, "
              f"{state['elapsed'] / 3600:.2f} h elapsed, best val minADE {state['best_val_minade']:.3f}")
    else:
        print(f"Fresh init (seed {SEED}).")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | bf16 autocast: {use_bf16} | "
          f"cumulative offsets: {model.cumulative_offsets} | anchors: {ANCHOR_SPACE}")

    def _save(path, with_optimizer=False, extra=None):
        payload = {"model_state_dict": model.state_dict(), "anchor_paths": anchor_paths,
                   "anchor_kf": {k: v.detach().cpu() for k, v in loss_bank["anchor_kf"].items()},
                   "config": run_config, "train_state": dict(state),
                   "anchor_group": (loss_bank["anchor_group"].detach().cpu()
                                    if config.ASSIGNMENT_MODE == "grouped" else None),
                   "epoch": state["epoch"], "loss": extra.get("train_loss", float("nan")) if extra else float("nan")}
        if with_optimizer:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if extra:
            payload.update(extra)
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)     # atomic: a crash mid-save never leaves a corrupt checkpoint

    rank_label = "gtop1" if hybrid else "atop1"
    reg_txt = (lambda x: f" reg=asg {x:.1%}") if hybrid else (lambda x: "")

    def validate(tag, use_learned_std, train_loss):
        val_m = compute_metrics(model, val_scenes, batch_size=EVAL_BATCH_SIZE)
        val_loss, val_atop1 = compute_val_loss(model, val_bank, loss_bank, use_learned_std=use_learned_std,
                                               use_bf16=use_bf16)
        oracle = oracle_minade(model, val_bank, use_bf16=use_bf16)
        improved = val_m["minADE"] < state["best_val_minade"]
        if improved:
            state["best_val_minade"] = val_m["minADE"]
            _save(best_checkpoint_path, extra={
                "train_loss": train_loss, "val_minADE": val_m["minADE"], "val_minFDE": val_m["minFDE"],
                "val_miss_rate": val_m["miss_rate"], "val_loss": val_loss, "val_oracle_minADE_20k": oracle,
                "val_p_top1": val_m["p_top1"], "val_top1_is_best": val_m["top1_is_best"],
                "val_offroad_rate": val_m["offroad_rate"], "val_gt_offroad_rate": val_m["gt_offroad_rate"],
                "val_assigned_top1": val_atop1})
        print(f"  VAL {tag}: minADE {val_m['minADE']:.3f} (best of {NUM_MODES}: {oracle:.3f}) "
              f"minFDE {val_m['minFDE']:.2f} MR {val_m['miss_rate']:.1%} "
              f"p_top1 {val_m['p_top1']:.2f} top1_is_best {val_m['top1_is_best']:.1%} "
              f"off-road {val_m['offroad_rate']:.1%} (GT {val_m['gt_offroad_rate']:.1%}) "
              f"{rank_label} {val_atop1:.1%} val_loss {val_loss:.4f}{'  (best)' if improved else ''}", flush=True)

    # ---- training loop ----
    steps_per_epoch = n_train // BATCH_SIZE     # last partial batch dropped
    start_wall = time.time() - state["elapsed"]
    elapsed = lambda: time.time() - start_wall
    last_ckpt = time.time()
    interrupted = False
    print(f"Training up to {TRAIN_TIME_BUDGET_SEC / 3600:.1f} h | {n_train} train samples, {steps_per_epoch} steps/epoch "
          f"at batch {BATCH_SIZE} | LR warmup {WARMUP_STEPS} steps to {PEAK_LR:.0e}, cosine to {MIN_LR:.0e} over the budget | "
          f"sigma=1 for {NLL_WARMUP_STEPS} steps | conf target: {conf_desc} | "
          f"loss = {config.NLL_WEIGHT}*nll + {config.CONF_WEIGHT}*conf | checkpoints -> {ckpt_dir}", flush=True)

    zero = torch.zeros((), device=device)
    model.train()
    try:
        while elapsed() < TRAIN_TIME_BUDGET_SEC:
            epoch = state["epoch"] + 1
            perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(SEED + epoch)).to(device)
            ep_sum = torch.zeros(3, device=device); ep_n = 0
            # window: loss, nll, conf, p_top1, KL, target entropy, target winner mass, rank top-1, reg==assigned
            win = torch.zeros(9, device=device); win_n = 0; win_t = time.time()
            win_gn = torch.zeros((), device=device)

            for b in range(state["batch_pos"], steps_per_epoch):
                if elapsed() >= TRAIN_TIME_BUDGET_SEC:
                    break
                idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
                f, v, fx, fv, rx, rt, rv, h = bank_batch(train_bank, idx)
                use_learned_std = state["global_step"] >= NLL_WARMUP_STEPS

                lr = lr_schedule(state["global_step"], elapsed())
                for g in optimizer.param_groups:
                    g["lr"] = lr

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    pt, pc, pl = model(f, v, rx, rt, rv, h)
                loss, nll, conf, stats = gaussian_nll_loss(pt, pc, pl, fx, fv, loss_bank,
                                                           use_learned_std=use_learned_std)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

                state["global_step"] += 1
                state["batch_pos"] = b + 1
                ld = loss.detach()
                ep_sum += torch.stack([ld, nll, conf]); ep_n += 1
                win += torch.stack([ld, nll, conf, stats["p_top1"], conf - stats["tgt_entropy"],
                                    stats["tgt_entropy"], stats["tgt_win"],
                                    stats.get("rank_top1", zero), stats.get("reg_is_assigned", zero)]); win_n += 1
                win_gn += gn.detach()

                if state["global_step"] % LOG_EVERY_STEPS == 0:
                    w = (win / win_n).tolist()
                    if not math.isfinite(w[0]):
                        raise RuntimeError(f"Non-finite loss near step {state['global_step']}: disable bf16 and resume.")
                    sps = win_n * BATCH_SIZE / (time.time() - win_t)
                    proj = TRAIN_TIME_BUDGET_SEC * sps / n_train
                    print(f"step {state['global_step']} (ep {epoch} {b + 1}/{steps_per_epoch}, {elapsed() / 60:.1f} min) "
                          f"loss {w[0]:.3f} nll {w[1]:.3f} conf {w[2]:.3f} (KL {w[4]:.3f}, target H {w[5]:.2f} win {w[6]:.2f}) "
                          f"p_top1 {w[3]:.2f} {rank_label} {w[7]:.1%}{reg_txt(w[8])} grad {float(win_gn) / win_n:.2f} "
                          f"lr {lr:.2e} | {sps:.0f} samples/s, "
                          f"~{proj:.1f} epochs in budget{'' if use_learned_std else ' [sigma=1]'}", flush=True)
                    win.zero_(); win_gn.zero_(); win_n = 0; win_t = time.time()

                if time.time() - last_ckpt >= CHECKPOINT_EVERY_SEC:
                    state["elapsed"] = elapsed()
                    _save(checkpoint_path, with_optimizer=True)
                    last_ckpt = time.time()
            else:
                # full epoch completed
                state["epoch"] = epoch
                state["batch_pos"] = 0
                state["elapsed"] = elapsed()
                e = (ep_sum / max(ep_n, 1)).tolist()
                print(f"Epoch {epoch} done ({elapsed() / 60:.1f} min): loss {e[0]:.4f} nll {e[1]:.4f} conf {e[2]:.4f}",
                      flush=True)
                validate(f"epoch {epoch}", state["global_step"] >= NLL_WARMUP_STEPS, e[0])
                _save(checkpoint_path, with_optimizer=True)
                last_ckpt = time.time()
                continue
            break   # inner loop broke: time budget reached mid-epoch
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted.")

    state["elapsed"] = elapsed()
    _save(checkpoint_path, with_optimizer=True)
    if not interrupted and state["batch_pos"] > 0:
        validate(f"final partial epoch {state['epoch'] + 1} ({state['batch_pos']}/{steps_per_epoch} batches)",
                 state["global_step"] >= NLL_WARMUP_STEPS, float("nan"))

    print(f"\nDone{' (interrupted -- pass resume_run_dir to continue)' if interrupted else ''}: "
          f"{state['global_step']} steps, {state['epoch']} full epochs, {state['elapsed'] / 60:.1f} min, "
          f"best val minADE {state['best_val_minade']:.3f} m")
    print(f"Best checkpoint (by val minADE): {best_checkpoint_path}")
    return best_checkpoint_path
