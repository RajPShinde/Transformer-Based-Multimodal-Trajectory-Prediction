"""Transformer encoder over agent-history and map / traffic-light tokens, then an anchor-conditioned mode decoder
that predicts, per mode, a trajectory (cumulative offsets from its anchor), a per-step sigma and a confidence."""

import math

import torch
import torch.nn as nn

from config import (AGENT_FEAT_DIM, CONF_HEAD, CUMULATIVE_OFFSETS, D_MODEL, DECODER_FF_DIM, DECODER_ROUNDS, DROPOUT,
                    FF_DIM, FUTURE_STEPS, HIST_STEPS, LOG_STD_MAX, LOG_STD_MIN, MAX_AGENTS, N_HEADS, N_LAYERS,
                    NUM_MODES, NUM_ROAD_BUCKETS, POS_SCALE_M, ROAD_FEAT_DIM)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        return x + self.pe[: x.size(0)].unsqueeze(1)


class TrajectoryTransformer(nn.Module):
    def __init__(
        self,
        feat_dim=AGENT_FEAT_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        decoder_rounds=DECODER_ROUNDS,
        ff_dim=FF_DIM,
        decoder_ff_dim=DECODER_FF_DIM,
        num_modes=NUM_MODES,
        max_agents=MAX_AGENTS,
        past_steps=HIST_STEPS,
        future_steps=FUTURE_STEPS,
        num_road_buckets=NUM_ROAD_BUCKETS,
        dropout=DROPOUT,
        anchor_paths=None,   # [num_modes, future_steps, 2] fixed anchor paths, agent frame
        cumulative_offsets=CUMULATIVE_OFFSETS,
        conf_head_type=CONF_HEAD,
        anchor_group=None,   # [num_modes] long, maneuver group of each slot (grouped anchors only)
        num_groups=0,        # > 0 adds a learned group embedding to each mode query
    ):
        super().__init__()
        self.max_agents = max_agents
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.cumulative_offsets = cumulative_offsets
        self.conf_head_type = conf_head_type

        # Agent branch
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.agent_embed = nn.Embedding(max_agents, d_model)  # marks which agent is which (target vs neighbors)
        self.pos_encoding = PositionalEncoding(d_model, max_len=past_steps)

        # Road graph + traffic-light branch
        self.road_proj = nn.Linear(ROAD_FEAT_DIM, d_model)
        self.road_type_embed = nn.Embedding(num_road_buckets, d_model)

        # Modality tag: 0 = agent history, 1 = road / traffic light
        self.modality_embed = nn.Embedding(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout,
            batch_first=False, norm_first=True,  # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Mode queries conditioned on anchor endpoint (and maneuver group); small init lets the anchor lead
        self.mode_queries = nn.Parameter(torch.randn(num_modes, d_model) * 0.02)
        self.anchor_proj = nn.Linear(2, d_model)  # anchor endpoint (agent frame, /POS_SCALE_M) -> query space
        self.num_groups = num_groups
        if num_groups > 0:
            self.group_embed = nn.Embedding(num_groups, d_model)
            nn.init.normal_(self.group_embed.weight, std=0.02)     # small, like the mode queries
            if anchor_group is None:
                anchor_group = torch.zeros(num_modes, dtype=torch.long)
            self.register_buffer("anchor_group", torch.as_tensor(anchor_group).long().clone())
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=decoder_ff_dim, dropout=dropout,
            batch_first=False, norm_first=True,  # Pre-LN
        )
        self.mode_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_rounds)

        # Per-timestep decode: future positional encoding + shared MLP -> (dx, dy, log_std)
        self.future_pos_encoding = PositionalEncoding(d_model, max_len=future_steps)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(d_model, decoder_ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_ff_dim, 3),  # (dx, dy, log_std)
        )
        if cumulative_offsets:
            with torch.no_grad():   # start exactly on the anchors
                self.decoder_mlp[-1].weight[:2].zero_()
                self.decoder_mlp[-1].bias[:2].zero_()

        if conf_head_type == "mlp":
            self.conf_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                           nn.ReLU(), nn.Linear(d_model, 1))
        elif conf_head_type == "linear":
            self.conf_head = nn.Linear(d_model, 1)
        else:
            raise ValueError(f"Unknown conf_head_type: {conf_head_type}")

        if anchor_paths is None:
            anchor_paths = torch.zeros(num_modes, future_steps, 2)
        self.register_buffer("anchor_paths", anchor_paths)

    def set_anchors(self, anchor_paths):
        """Overwrite the fixed anchor trajectories in place."""
        self.anchor_paths.copy_(anchor_paths.to(self.anchor_paths.device))

    def _rotated_anchors(self, heading):
        """Rotate the anchor bank [M, T, 2] by each sample's heading [B, 2] -> [B, M, T, 2].
        Identity for agent-frame inputs (heading = [1, 0])."""
        theta = torch.atan2(heading[:, 1], heading[:, 0])  # [B]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)  # [B]

        anchors = self.anchor_paths.unsqueeze(0)            # [1, M, T, 2]
        ax, ay = anchors[..., 0], anchors[..., 1]            # [1, M, T]

        cos_t = cos_t.view(-1, 1, 1)  # [B, 1, 1]
        sin_t = sin_t.view(-1, 1, 1)

        rx = ax * cos_t - ay * sin_t  # [B, M, T]
        ry = ax * sin_t + ay * cos_t  # [B, M, T]
        return torch.stack([rx, ry], dim=-1)  # [B, M, T, 2]

    def forward(self, agent_feats, agent_valid, road_xy=None, road_type=None, road_valid=None, heading=None):
        # agent_feats [B, A, T, F], agent_valid [B, A, T], road_xy [B, R, ROAD_FEAT_DIM], heading [B, 2] or None
        B, A, T, _ = agent_feats.shape
        device = agent_feats.device

        x = self.input_proj(agent_feats)  # [B, A, T, D]
        agent_ids = torch.arange(A, device=device)
        x = x + self.agent_embed(agent_ids).view(1, A, 1, -1)

        # Positional encoding along the time axis
        x = x.permute(2, 0, 1, 3).reshape(T, B * A, -1)
        x = self.pos_encoding(x)
        x = x.reshape(T, B, A, -1).permute(1, 2, 0, 3).reshape(B, A * T, -1)
        x = x + self.modality_embed(torch.zeros(1, dtype=torch.long, device=device)).view(1, 1, -1)

        agent_padding_mask = agent_valid.reshape(B, A * T) == 0  # True = ignore this token

        if road_xy is not None and road_xy.shape[1] > 0:
            r = self.road_proj(road_xy) + self.road_type_embed(road_type)  # [B, R, D]
            r = r + self.modality_embed(torch.ones(1, dtype=torch.long, device=device)).view(1, 1, -1)
            road_padding_mask = road_valid == 0  # [B, R]

            tokens = torch.cat([x, r], dim=1)                       # [B, A*T + R, D]
            padding_mask = torch.cat([agent_padding_mask, road_padding_mask], dim=1)
        else:
            tokens = x
            padding_mask = agent_padding_mask

        tokens = tokens.permute(1, 0, 2)  # [seq_len, B, D]: encoder expects seq-first here
        memory = self.encoder(tokens, src_key_padding_mask=padding_mask)  # [seq_len, B, D]

        # Mode decoding: each query conditioned on its anchor endpoint
        anchor_ends = self.anchor_paths[:, -1, :]                                   # [M, 2]
        mode_q = self.mode_queries + self.anchor_proj(anchor_ends / POS_SCALE_M)    # [M, D]
        if self.num_groups > 0:
            mode_q = mode_q + self.group_embed(self.anchor_group)                    # shared per-group term
        queries = mode_q.unsqueeze(1).repeat(1, B, 1)  # [M, B, D]
        mode_out = self.mode_decoder(queries, memory, memory_key_padding_mask=padding_mask)  # [M, B, D]

        # Broadcast each mode across the horizon, add future positional encoding, decode per timestep
        future_pe = self.future_pos_encoding.pe[:self.future_steps]          # [T, D]
        decoder_in = mode_out.unsqueeze(2) + future_pe.view(1, 1, self.future_steps, -1)  # [M, B, T, D]
        hidden = self.decoder_mlp[:-1](decoder_in)  # [M, B, T, DECODER_FF_DIM] (bf16 under autocast)

        # Output layer, confidence head and anchor addition in fp32 for coordinate precision
        with torch.autocast(device_type=device.type, enabled=False):
            raw = self.decoder_mlp[-1](hidden.float())  # [M, B, T, 3]
            raw = raw.permute(1, 0, 2, 3)               # [B, M, T, 3]

            if self.cumulative_offsets:
                offset = torch.cumsum(raw[..., :2], dim=2)   # per-step deltas, integrated over time
            else:
                offset = raw[..., :2]                        # direct per-step offsets
            log_std = raw[..., 2].clamp(LOG_STD_MIN, LOG_STD_MAX)    # [B, M, T]

            if heading is not None:
                anchors = self._rotated_anchors(heading.float())     # [B, M, T, 2]
            else:
                anchors = self.anchor_paths.unsqueeze(0)              # [1, M, T, 2]
            trajs = anchors + offset  # anchor-relative regression

            confs = self.conf_head(mode_out.float()).squeeze(-1).permute(1, 0)  # [B, M]

        return trajs, confs, log_std


def load_checkpoint(path, device, session_anchors=None):
    """Rebuilds the model from a checkpoint with its own anchors, head type and group settings.

    Returns (model, ckpt, extras). extras holds what the loss needs to keep training or computing val loss:
    anchor_kf, and for grouped runs group_anchors / anchor_group (all on `device`, else None).
    session_anchors: optional anchors fit in this session, compared against the checkpoint's."""
    print(f"Loading {path}")
    ckpt = torch.load(path, map_location=device)
    ckpt_cfg = ckpt.get("config", {})
    sd = ckpt["model_state_dict"]
    conf_head_type = "linear" if "conf_head.weight" in sd else "mlp"
    num_groups = sd["group_embed.weight"].shape[0] if "group_embed.weight" in sd else 0
    ckpt_group = ckpt.get("anchor_group")
    if ckpt_group is None and "anchor_group" in sd:
        ckpt_group = sd["anchor_group"]

    ckpt_anchors = ckpt.get("anchor_paths")
    if ckpt_anchors is None:
        if session_anchors is None:
            raise RuntimeError("Checkpoint has no anchor_paths; pass session_anchors.")
        ckpt_anchors = session_anchors
        print("WARNING: checkpoint has no anchor_paths; using refitted anchors.")

    model = TrajectoryTransformer(anchor_paths=ckpt_anchors,
                                  cumulative_offsets=ckpt_cfg.get("cumulative_offsets", False),
                                  conf_head_type=conf_head_type,
                                  anchor_group=ckpt_group, num_groups=num_groups).to(device)
    model.load_state_dict(sd)
    model.set_anchors(ckpt_anchors.to(device))
    model.to(device)

    grouped = ckpt_cfg.get("assignment_mode") == "grouped"
    extras = {
        "anchor_kf": {k: v.to(device) for k, v in ckpt["anchor_kf"].items()} if ckpt.get("anchor_kf") is not None else None,
        "group_anchors": ckpt_anchors.to(device) if grouped else None,
        "anchor_group": torch.as_tensor(ckpt_group).long().to(device) if grouped else None,
    }

    if session_anchors is not None:
        match = torch.allclose(ckpt_anchors.cpu(), session_anchors.cpu(), atol=1e-4)
        maxdiff = (ckpt_anchors.cpu() - session_anchors.cpu()).abs().max().item()
        print(f"Session anchors match checkpoint anchors: {match} (max abs diff {maxdiff:.4g})")

    print(f"Loaded best checkpoint: epoch {ckpt['epoch']}, train loss {ckpt.get('loss', float('nan')):.4f}, "
          f"val minADE {ckpt.get('val_minADE', float('nan')):.2f}m, val minFDE {ckpt.get('val_minFDE', float('nan')):.2f}m, "
          f"val_loss {ckpt.get('val_loss', float('nan')):.4f} | cumulative offsets: {model.cumulative_offsets} | "
          f"conf head: {model.conf_head_type} | anchors: {'grouped' if grouped else ckpt_cfg.get('anchor_space', '?')}, "
          f"group query embedding: {num_groups > 0}")
    return model, ckpt, extras
