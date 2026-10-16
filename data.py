"""Waymo tf.Example parsing, scene extraction (road graph, traffic lights, agents) and agent-frame model inputs."""

import math

import numpy as np
import torch

from config import (AGENT_FEAT_DIM, HIST_STEPS, MAX_AGENTS, MAX_ROAD_POINTS, MAX_TARGETS_PER_SCENE,
                    MAX_TRAFFIC_LIGHTS, NUM_TL_STATES, PAST_STEPS, FUTURE_STEPS, POS_SCALE_M, ROAD_FEAT_DIM,
                    ROAD_FORWARD_OFFSET_M, ROAD_POINT_STRIDE, ROAD_RADIUS_M, TL_BUCKET_OFFSET, VELOCITY_SCALE_MPS)

# Waymo object type codes
TYPE_NAMES = {0: "unset", 1: "vehicle", 2: "pedestrian", 3: "cyclist", 4: "other"}
AGENT_TYPE_SLOT = {1: 0, 2: 1, 3: 2}   # one-hot slot within the agent feature vector

# Waymo roadgraph type codes -> buckets (1-3 lanes, 6-13 lines, 15-16 edges, 17-20 point features)
ROAD_TYPE_BUCKET = {t: 1 for t in (1, 2, 3)}
ROAD_TYPE_BUCKET.update({t: 2 for t in range(6, 14)})
ROAD_TYPE_BUCKET.update({t: 3 for t in (15, 16)})
ROAD_TYPE_BUCKET[17] = 4  # stop sign
ROAD_TYPE_BUCKET[18] = 5  # crosswalk
ROAD_TYPE_BUCKET[19] = 6  # speed bump
ROAD_TYPE_BUCKET[20] = 7  # driveway

ROAD_TYPE_LUT = np.zeros(64, dtype=np.int64)   # vectorized type -> bucket lookup
for _t, _b in ROAD_TYPE_BUCKET.items():
    ROAD_TYPE_LUT[_t] = _b


# ---- tf.Example parsing ----

def feature_spec():
    """The subset of Waymo tf.Example features this project reads."""
    import tensorflow as tf
    spec = {
        "state/type": tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        "state/past/x": tf.io.FixedLenFeature([128, PAST_STEPS], tf.float32, default_value=None),
        "state/past/y": tf.io.FixedLenFeature([128, PAST_STEPS], tf.float32, default_value=None),
        "state/past/velocity_x": tf.io.FixedLenFeature([128, PAST_STEPS], tf.float32, default_value=None),
        "state/past/velocity_y": tf.io.FixedLenFeature([128, PAST_STEPS], tf.float32, default_value=None),
        "state/past/bbox_yaw": tf.io.FixedLenFeature([128, PAST_STEPS], tf.float32, default_value=None),
        "state/past/valid": tf.io.FixedLenFeature([128, PAST_STEPS], tf.int64, default_value=None),
        "state/current/x": tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/current/y": tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/current/velocity_x": tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/current/velocity_y": tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/current/bbox_yaw": tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        "state/current/valid": tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
        "state/future/x": tf.io.FixedLenFeature([128, FUTURE_STEPS], tf.float32, default_value=None),
        "state/future/y": tf.io.FixedLenFeature([128, FUTURE_STEPS], tf.float32, default_value=None),
        "state/future/valid": tf.io.FixedLenFeature([128, FUTURE_STEPS], tf.int64, default_value=None),
        "state/tracks_to_predict": tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        "roadgraph_samples/xyz": tf.io.FixedLenFeature([30000, 3], tf.float32, default_value=None),
        "roadgraph_samples/dir": tf.io.FixedLenFeature([30000, 3], tf.float32, default_value=None),
        "roadgraph_samples/type": tf.io.FixedLenFeature([30000, 1], tf.int64, default_value=None),
        "roadgraph_samples/id": tf.io.FixedLenFeature([30000, 1], tf.int64, default_value=None),
        "roadgraph_samples/valid": tf.io.FixedLenFeature([30000, 1], tf.int64, default_value=None),
    }
    # Traffic lights: [time, light] grids (10 past steps + current)
    for k, dt in (("state", tf.int64), ("valid", tf.int64), ("x", tf.float32), ("y", tf.float32)):
        spec[f"traffic_light_state/past/{k}"] = tf.io.FixedLenFeature([PAST_STEPS, MAX_TRAFFIC_LIGHTS], dt, default_value=None)
        spec[f"traffic_light_state/current/{k}"] = tf.io.FixedLenFeature([1, MAX_TRAFFIC_LIGHTS], dt, default_value=None)
    return spec


def load_dataset(paths):
    """Loads one shard path or a list of shard paths as a single parsed dataset."""
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")   # keep TensorFlow off the GPU; PyTorch owns it
    except RuntimeError:
        pass                                       # TF already initialized (an earlier call did this)
    if isinstance(paths, str):
        paths = [paths]
    spec = feature_spec()
    return tf.data.TFRecordDataset(paths, compression_type="").map(lambda s: tf.io.parse_single_example(s, spec))


# ---- Scene extraction ----

def extract_road_graph(parsed_scene, origin_xy, radius=ROAD_RADIUS_M, max_points=MAX_ROAD_POINTS,
                       square=False, forward_dir=None, forward_offset=0.0, stride=1):
    """Crops the road graph. Returns (xy, bucket, feature_id, dir_xy) in world coordinates.

    stride: keep every `stride`-th point per polyline (plus each polyline's
      first point and all stop signs / speed bumps).
    forward_dir / forward_offset: shift the crop center ahead of the agent.
    square=True: square crop (visualization only).
    Beyond max_points, the points farthest from the crop center are dropped.
    """
    xyz = parsed_scene["roadgraph_samples/xyz"].numpy()
    rdir = parsed_scene["roadgraph_samples/dir"].numpy()
    rtype = parsed_scene["roadgraph_samples/type"].numpy().reshape(-1).astype(int)
    rid = parsed_scene["roadgraph_samples/id"].numpy().reshape(-1).astype(int)
    rvalid = parsed_scene["roadgraph_samples/valid"].numpy().reshape(-1).astype(bool)

    empty = (np.zeros((0, 2), np.float32), np.zeros((0,), np.int64),
             np.zeros((0,), np.int64), np.zeros((0, 2), np.float32))

    n = len(rtype)
    if stride > 1:
        starts = np.r_[True, rid[1:] != rid[:-1]]
        seg_start = np.maximum.accumulate(np.where(starts, np.arange(n), 0))
        pos_in_seg = np.arange(n) - seg_start
        point_like = np.isin(rtype, (17, 19))
        keep = rvalid & ((pos_in_seg % stride == 0) | point_like)
    else:
        keep = rvalid
    idx = np.where(keep)[0]
    if len(idx) == 0:
        return empty

    xy, types, ids, dirs = xyz[idx, :2], rtype[idx], rid[idx], rdir[idx, :2]

    center = np.asarray(origin_xy, dtype=np.float64)
    if forward_dir is not None and forward_offset != 0.0:
        center = center + forward_offset * np.asarray(forward_dir, dtype=np.float64)
    offset = xy - center
    dist = np.abs(offset).max(axis=1) if square else np.linalg.norm(offset, axis=1)

    within = np.where(dist <= radius)[0]
    if len(within) > max_points:
        within = within[np.argsort(dist[within])[:max_points]]
        within.sort()  # restore sequential-along-polyline order
    if len(within) == 0:
        return empty

    bucket = ROAD_TYPE_LUT[np.clip(types[within], 0, len(ROAD_TYPE_LUT) - 1)]
    return (xy[within].astype(np.float32), bucket.astype(np.int64),
            ids[within].astype(np.int64), dirs[within].astype(np.float32))


def extract_traffic_lights(parsed_scene):
    """Lights valid at t=0 -> (xy [K, 2] world, bucket [K], time_in_state [K]).
    time_in_state is the fraction of the 1.1s history the light has held its
    current state (1.0 = unchanged, small = just changed)."""
    st = np.concatenate([parsed_scene["traffic_light_state/past/state"].numpy(),
                         parsed_scene["traffic_light_state/current/state"].numpy()], axis=0)   # [11, L]
    va = np.concatenate([parsed_scene["traffic_light_state/past/valid"].numpy(),
                         parsed_scene["traffic_light_state/current/valid"].numpy()], axis=0) > 0
    cx = parsed_scene["traffic_light_state/current/x"].numpy()[0]
    cy = parsed_scene["traffic_light_state/current/y"].numpy()[0]

    lights = np.where(va[-1])[0]
    if len(lights) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.int64), np.zeros((0,), np.float32)

    cur = st[-1, lights]
    same = va[:, lights] & (st[:, lights] == cur[None, :])            # [11, K]
    # length of the unbroken run of `same` ending at t=0
    broken = ~same[::-1]                                              # reversed time
    run = np.where(broken.any(axis=0), broken.argmax(axis=0), HIST_STEPS)
    time_in_state = (run / HIST_STEPS).astype(np.float32)

    bucket = TL_BUCKET_OFFSET + np.clip(cur, 0, NUM_TL_STATES - 1)
    xy = np.stack([cx[lights], cy[lights]], axis=1).astype(np.float32)
    return xy, bucket.astype(np.int64), time_in_state


def _hist(parsed_scene, key):
    """[128, HIST_STEPS]: the 10 past steps followed by current (last column = t=0)."""
    past = parsed_scene[f"state/past/{key}"].numpy()
    cur = parsed_scene[f"state/current/{key}"].numpy()
    return np.concatenate([past, cur], axis=1)


def extract_scene_agents(parsed_scene, max_agents=MAX_AGENTS, max_targets=MAX_TARGETS_PER_SCENE):
    """One sample per tracks_to_predict agent (up to max_targets): the target plus its nearest neighbors,
    road-graph crop and traffic lights, all in world coordinates."""
    x, y = _hist(parsed_scene, "x"), _hist(parsed_scene, "y")
    vx, vy = _hist(parsed_scene, "velocity_x"), _hist(parsed_scene, "velocity_y")
    yaw = _hist(parsed_scene, "bbox_yaw")
    valid = _hist(parsed_scene, "valid")
    obj_type = parsed_scene["state/type"].numpy().astype(int)
    fut_x = parsed_scene["state/future/x"].numpy()
    fut_y = parsed_scene["state/future/y"].numpy()
    fut_valid = parsed_scene["state/future/valid"].numpy()
    to_predict = parsed_scene["state/tracks_to_predict"].numpy()
    tl_xy, tl_type, tl_time = extract_traffic_lights(parsed_scene)

    # each agent's last valid history position, used to rank neighbors
    last_t = np.where(valid > 0, np.arange(HIST_STEPS), -1).max(axis=1)
    has_hist = last_t >= 0
    rows = np.arange(len(last_t))
    safe_t = np.maximum(last_t, 0)
    last_xy = np.stack([x[rows, safe_t], y[rows, safe_t]], axis=1)

    samples = []
    for target_idx in np.where(to_predict == 1)[0]:
        if len(samples) >= max_targets:
            break
        # target needs a valid current state and some future
        if valid[target_idx, -1] == 0 or fut_valid[target_idx].sum() == 0:
            continue

        origin = np.array([x[target_idx, -1], y[target_idx, -1]], dtype=np.float32)
        yaw0 = float(yaw[target_idx, -1])

        others = np.where(has_hist)[0]
        others = others[others != target_idx]
        dists = np.linalg.norm(last_xy[others] - origin, axis=1)
        nearest = others[np.argsort(dists)[: max_agents - 1]]
        selected = np.concatenate([[target_idx], nearest]).astype(int)

        road_xy, road_type, road_id, road_dir = extract_road_graph(
            parsed_scene, origin,
            forward_dir=(math.cos(yaw0), math.sin(yaw0)), forward_offset=ROAD_FORWARD_OFFSET_M,
            stride=ROAD_POINT_STRIDE,
        )

        samples.append({
            "target_idx": 0,               # target is always row 0
            "agent_ids": selected,
            "type": obj_type[selected],
            "x": x[selected], "y": y[selected],                # [A, HIST_STEPS], world coords
            "vx": vx[selected], "vy": vy[selected],
            "bbox_yaw": yaw[selected],
            "valid": valid[selected],
            "future_x": fut_x[selected], "future_y": fut_y[selected],
            "future_valid": fut_valid[selected],
            "origin": origin,              # target's current position (world)
            "yaw0": yaw0,                  # target's current yaw (world) -> agent frame +x
            "road_xy": road_xy, "road_type": road_type,        # world coords, cropped
            "road_id": road_id, "road_dir": road_dir,
            "tl_xy": tl_xy, "tl_type": tl_type, "tl_time": tl_time,   # world coords
        })
    return samples


# ---- Agent-frame model inputs (origin = target's current position, +x = its heading) ----

def to_agent_frame(xy, origin, yaw):
    """World [..., 2] points -> agent frame (translate to origin, rotate by -yaw)."""
    d = np.asarray(xy, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], axis=-1).astype(np.float32)


def rotate_to_agent_frame(v, yaw):
    """World [..., 2] direction/velocity vectors -> agent frame (rotation only)."""
    v = np.asarray(v, dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([v[..., 0] * c + v[..., 1] * s, -v[..., 0] * s + v[..., 1] * c], axis=-1).astype(np.float32)


def build_model_inputs(scene, max_agents=MAX_AGENTS, hist_steps=HIST_STEPS, max_road_points=MAX_ROAD_POINTS,
                       max_lights=MAX_TRAFFIC_LIGHTS):
    """Converts one extracted sample into padded tensors in the target's agent frame."""
    origin, yaw0 = scene["origin"], scene["yaw0"]
    num_agents = scene["x"].shape[0]

    feats = np.zeros((max_agents, hist_steps, AGENT_FEAT_DIM), dtype=np.float32)
    valid = np.zeros((max_agents, hist_steps), dtype=np.float32)

    for i in range(min(num_agents, max_agents)):
        m = scene["valid"][i] > 0
        pos = to_agent_frame(np.stack([scene["x"][i], scene["y"][i]], axis=-1), origin, yaw0)
        vel = rotate_to_agent_frame(np.stack([scene["vx"][i], scene["vy"][i]], axis=-1), yaw0)
        dyaw = scene["bbox_yaw"][i] - yaw0
        feats[i, :, 0:2] = pos / POS_SCALE_M
        feats[i, :, 2:4] = vel / VELOCITY_SCALE_MPS
        feats[i, :, 4] = np.cos(dyaw)
        feats[i, :, 5] = np.sin(dyaw)
        slot = AGENT_TYPE_SLOT.get(int(scene["type"][i]))
        if slot is not None:
            feats[i, :, 6 + slot] = 1.0
        feats[i, ~m] = 0.0          # zero out padding steps (-1 sentinels), they're masked anyway
        valid[i] = m

    future_valid = scene["future_valid"][0] > 0
    future_xy = to_agent_frame(np.stack([scene["future_x"][0], scene["future_y"][0]], axis=-1), origin, yaw0)
    future_xy[~future_valid] = 0.0

    road_xy_raw = scene.get("road_xy", np.zeros((0, 2), dtype=np.float32))
    road_dir_raw = scene.get("road_dir", np.zeros((0, 2), dtype=np.float32))
    road_type_raw = scene.get("road_type", np.zeros((0,), dtype=np.int64))
    road_id_raw = scene.get("road_id", np.zeros((0,), dtype=np.int64))
    n_road = min(len(road_xy_raw), max_road_points)

    n_tokens = max_road_points + max_lights
    road_feats = np.zeros((n_tokens, ROAD_FEAT_DIM), dtype=np.float32)
    road_type = np.zeros((n_tokens,), dtype=np.int64)
    road_valid = np.zeros((n_tokens,), dtype=np.float32)
    road_id = np.full((n_tokens,), -1, dtype=np.int64)

    if n_road > 0:
        road_feats[:n_road, 0:2] = to_agent_frame(road_xy_raw[:n_road], origin, yaw0) / POS_SCALE_M
        road_feats[:n_road, 2:4] = rotate_to_agent_frame(road_dir_raw[:n_road], yaw0)
        road_type[:n_road] = road_type_raw[:n_road]
        road_valid[:n_road] = 1.0
        road_id[:n_road] = road_id_raw[:n_road]

    # traffic lights occupy the slot range [max_road_points, max_road_points + max_lights)
    tl_xy_raw = scene.get("tl_xy", np.zeros((0, 2), dtype=np.float32))
    n_tl = min(len(tl_xy_raw), max_lights)
    if n_tl > 0:
        sl = slice(max_road_points, max_road_points + n_tl)
        road_feats[sl, 0:2] = to_agent_frame(tl_xy_raw[:n_tl], origin, yaw0) / POS_SCALE_M
        road_feats[sl, 4] = scene["tl_time"][:n_tl]
        road_type[sl] = scene["tl_type"][:n_tl]
        road_valid[sl] = 1.0

    return {
        "agent_feats": torch.from_numpy(feats),              # [max_agents, HIST_STEPS, AGENT_FEAT_DIM]
        "agent_valid": torch.from_numpy(valid),              # [max_agents, HIST_STEPS]
        "future_xy": torch.from_numpy(future_xy),            # [FUTURE_STEPS, 2], meters, agent frame
        "future_valid": torch.from_numpy(future_valid.astype(np.float32)),
        "road_xy": torch.from_numpy(road_feats),             # [MAX_ROAD_TOKENS, 5], map points + traffic lights
        "road_type": torch.from_numpy(road_type),            # [MAX_ROAD_TOKENS]
        "road_valid": torch.from_numpy(road_valid),          # [MAX_ROAD_TOKENS]
        "road_id": torch.from_numpy(road_id),                # [MAX_ROAD_TOKENS], viz only
        "heading": torch.tensor([1.0, 0.0]),                 # agent frame: forward is +x, anchor rotation = identity
        "origin": origin,                                    # world, for viz only
        "yaw0": yaw0,                                        # world, for viz only
    }


def target_type(sample):
    """Target's agent type slot from its one-hot: 0 vehicle, 1 pedestrian, 2 cyclist, -1 unknown."""
    oh = sample["agent_feats"][0, -1, 6:9].float().numpy()
    return int(np.argmax(oh)) if oh.sum() > 0 else -1


def collate_scenes(batch):
    """Stacks samples into the model's 8-tuple, casting compactly stored tensors to model dtypes."""
    feats = torch.stack([s["agent_feats"] for s in batch]).float()
    valid = torch.stack([s["agent_valid"] for s in batch]).float()
    future_xy = torch.stack([s["future_xy"] for s in batch]).float()
    future_valid = torch.stack([s["future_valid"] for s in batch]).float()
    road_xy = torch.stack([s["road_xy"] for s in batch]).float()
    road_type = torch.stack([s["road_type"] for s in batch]).long()
    road_valid = torch.stack([s["road_valid"] for s in batch]).float()
    heading = torch.stack([s["heading"] for s in batch]).float()  # [B, 2], always [1, 0] in the agent frame
    return feats, valid, future_xy, future_valid, road_xy, road_type, road_valid, heading
