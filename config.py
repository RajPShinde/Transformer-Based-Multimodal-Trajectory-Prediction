"""All settings in one place. The defaults are the exact values that trained checkpoints/trajectory_transformer_best.pt."""

# ---- Data layout (changing these changes tensor shapes; re-run prep after editing) ----
MAX_AGENTS = 8                # agents kept per sample, including target
PAST_STEPS = 10               # state/past/* steps (1s before current, 0.1s resolution)
HIST_STEPS = PAST_STEPS + 1   # past + current (t=0): what the model sees
FUTURE_STEPS = 80             # 8s of future at 0.1s resolution

MAX_ROAD_POINTS = 768         # map-point budget per sample
ROAD_RADIUS_M = 120.0         # crop radius around the (forward-shifted) crop center
ROAD_POINT_STRIDE = 6         # keep every 6th sample along each polyline
ROAD_FORWARD_OFFSET_M = 30.0  # crop center this far ahead of the agent along its yaw

MAX_TRAFFIC_LIGHTS = 16       # Waymo stores up to 16 lights per scene
MAX_ROAD_TOKENS = MAX_ROAD_POINTS + MAX_TRAFFIC_LIGHTS   # map rows first, then light rows

MAX_TARGETS_PER_SCENE = 2     # targets per scene for viz; data prep keeps all

POS_SCALE_M = 50.0            # input positions divided by this (inputs only; outputs stay in meters)
VELOCITY_SCALE_MPS = 20.0
AGENT_FEAT_DIM = 9            # x, y, vx, vy, cos(dyaw), sin(dyaw), is_vehicle, is_pedestrian, is_cyclist
ROAD_FEAT_DIM = 5             # x, y, dir_x, dir_y, tl_time_in_state (0 for map points)

# Traffic-light bucket = TL_BUCKET_OFFSET + Waymo signal state (0-8)
TL_BUCKET_OFFSET = 8
NUM_TL_STATES = 9
NUM_ROAD_BUCKETS = TL_BUCKET_OFFSET + NUM_TL_STATES   # 17; bucket 0 = unset/other

# ---- Data prep (Waymo Open Motion v1.2 tf.Example shards -> compact tensors on Drive) ----
DRIVE_DIR = "/content/drive/MyDrive/waymo_trajectory"
SMOKE_TEST = False                     # True: 4-shard smoke set; False: all 600 shards
SEED = 42
if SMOKE_TEST:
    PREP_OUT_DIR = f"{DRIVE_DIR}/prepped_smoke"
    PREP_SHARD_IDS = [0, 1, 8, 9]      # 2 train, 1 val (id%10==8), 1 test (id%10==9)
else:
    PREP_OUT_DIR = f"{DRIVE_DIR}/prepped_600"
    PREP_SHARD_IDS = list(range(600))
DOWNLOAD_BATCH = 8                     # shards per parallel gsutil -m call (bounds local disk use)
DOWNLOAD_RETRIES = 3
SHARDS_PER_CHUNK = 20                  # shards accumulated per saved .pt chunk
MAX_STEP_SPEED_MPS = 45.0              # drop a sample if any consecutive valid future step exceeds this
GCS_PREFIX = "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/tf_example/training"
LOCAL_SHARD_DIR = "/content"           # raw shards are downloaded here, then deleted after prep

# ---- Model ----
NUM_MODES = 32         # one mode per anchor
EVAL_TOP_K = 6         # we score the best 6 modes (Waymo's convention)
D_MODEL = 192
N_HEADS = 4
N_LAYERS = 6           # encoder layers
DECODER_ROUNDS = 6     # decoder layers (modes attend to each other and to the scene)
FF_DIM = 768
DECODER_FF_DIM = 768   # decoder FFN width and per-timestep MLP width
DROPOUT = 0.1
CUMULATIVE_OFFSETS = True
CONF_HEAD = "mlp"      # "mlp" or "linear"
GROUP_QUERY_EMBED = True   # add a maneuver-group embedding to each mode query

LOG_STD_MIN = -1.6     # clamp the predicted uncertainty to this range
LOG_STD_MAX = 2.0      # (roughly sigma in [0.2, 7.4] metres)

# ---- Anchors ----
ANCHOR_FIT_N = 200_000
ANCHOR_MAX_DIST_M = 400.0       # ignore endpoints farther than this (glitches) when fitting
ANCHOR_SPACE = "shape"          # "shape" or "position" (base anchors only; training uses the grouped bank)
STATIONARY_LEN_M = 2.0
N_STRAIGHT_ANCHORS = 2
STRAIGHT_MAX_DEG = 10.0         # straight: final heading within this of +x ...
STRAIGHT_MAX_LAT_M = 1.5        # ... and within this lateral distance

# Maneuver groups from ground-truth geometry
GROUP_NAMES = ["stationary/reverse", "straight", "left turn", "right turn", "U-turn"]
GROUP_COLORS = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
TURN_DEG = 35.0
UTURN_DEG = 135.0
LANE_CHANGE_R_M = 1.75          # diagnostic only (dashed lines on the r histogram)
UTURN_MIN_LAT_M = 4.0           # a real U-turn swings sideways; reversing in place does not

GROUP_SLOTS = [2, 12, 8, 8, 2]                       # same order as GROUP_NAMES; must sum to NUM_MODES
GROUP_KMEANS_STANDARDIZE = False                     # scale k-means features per group (False = raw metres)

# ---- Loss ----
ASSIGNMENT_MODE = "grouped"      # "grouped", "trajectory", "endpoint", "anchor", or "soft"
SOFT_ASSIGN_TEMP_M = 0.5         # used only when ASSIGNMENT_MODE == "soft"
GROUPED_HYBRID = True            # grouped only: closest in-group slot regresses; soft in-group confidence
CONF_TARGET = "relative"         # "relative", "hard" or "fixed" (hybrid uses the in-group relative target)
REL_TAU_SCALE = 0.15
REL_TAU_BIAS = 0.2               # also the minimum tau (m)
REL_TAU_MAX = 2.0
CONF_TARGET_TEMP_M = 2.0         # only for CONF_TARGET == "fixed"
LABEL_SMOOTHING = 0.0            # only for CONF_TARGET == "hard"
SOFT_CONF_TARGET = CONF_TARGET != "hard"   # recorded in the run config
NLL_WEIGHT = 0.2                 # keeps the metre-scale NLL from swamping the confidence loss
CONF_WEIGHT = 1.0

# ---- Training ----
TRAIN_TIME_BUDGET_SEC = 10 * 60 * 60    # total across resumes
CHECKPOINT_EVERY_SEC = 10 * 60
LOG_EVERY_STEPS = 1000

BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
PEAK_LR = 3e-4          # peak learning rate
MIN_LR = 1e-6
WARMUP_STEPS = 3_000
NLL_WARMUP_STEPS = 15_000   # sigma held at 1 until here
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
ORACLE_VAL_N = 20_000       # val subset for the best-of-all-modes diagnostic

# ---- Evaluation ----
MISS_THRESHOLD_M = 2.0
OFFROAD_THRESH_M = 3.0
OFFROAD_STEP_STRIDE = 10   # check one point per second of the 8 s horizon
NMS_DIST_M = 2.5           # MTR's endpoint NMS threshold

# ---- Visualization ----
VIZ_ROAD_RADIUS_M = 150.0
VIZ_MAX_ROAD_POINTS = 2000
MAX_PLAUSIBLE_COORD_M = 400.0   # viz-only distance filter
N_VIZ_SAMPLES = 20     # turn-like scenes
N_VIZ_RANDOM = 20
N_VIZ_WORST = 20
PLOT_SHARD_IDS = [9, 19, 39, 89, 99, 109, 119, 159]   # held-out test shards to draw scenes from
VIEW_MARGIN_M = 10.0       # padding around the framed content
VIEW_MIN_HALF_M = 20.0     # never zoom in tighter than +-20 m
