import os

# ── 路径配置 ──────────────────────────────────────────────────────
DATA_ROOT         = r"H:\data\Article1\tiyu\data\taichi_frames"
KEYPOINTS_DIR     = r"H:\data\Article1\tiyu\data\taichi_keypoints"
MODEL_SAVE_DIR    = "./checkpoints"
LOG_DIR           = "./logs"

# ── MediaPipe Tasks 模型路径（0.10.x 必须使用 .task 文件） ────────
#    可选: pose_landmarker_lite.task / pose_landmarker_full.task
#          / pose_landmarker_heavy.task
MEDIAPIPE_MODEL_PATH = "./pose_landmarker_full.task"

# ── 特征维度 ──────────────────────────────────────────────────────
NUM_KEYPOINTS   = 33
FEATURE_DIM_POS = 99    # 33 × 3  位置
FEATURE_DIM_VEL = 99    # 33 × 3  一阶速度（帧间差分）
FEATURE_DIM     = FEATURE_DIM_POS + FEATURE_DIM_VEL  # 198

# D_MODEL / NHEAD / NUM_LAYERS 不变，input_proj 自动适配新 FEATURE_DIM

# ── 数据集固定参数 ────────────────────────────────────────────────
FRAMES_PER_SAMPLE = 103  # 每个样本文件夹固定 103 帧

# ── 序列参数（滑动窗口） ──────────────────────────────────────────
# SEQ_LEN < FRAMES_PER_SAMPLE 可产生多个训练窗口
# (103 - 64) // 8 + 1 = 5 个窗口/样本
SEQ_LEN           = 64
STRIDE            = 8

# ── 模型超参数 ────────────────────────────────────────────────────
D_MODEL           = 128
NHEAD             = 8
NUM_LAYERS        = 4
DIM_FEEDFORWARD   = 512
DROPOUT           = 0.15
NUM_CLASSES       = 24

# ── 训练超参数 ────────────────────────────────────────────────────
BATCH_SIZE        = 32
LEARNING_RATE     = 1e-3
WEIGHT_DECAY      = 1e-4
EPOCHS            = 120
TRAIN_RATIO       = 0.8

# ── 太极24式名称 ──────────────────────────────────────────────────
ACTION_NAMES = {
    1:  "01-起势",        2:  "02-左右野马分鬃",
    3:  "03-白鹤亮翅",    4:  "04-左右搂膝拗步",
    5:  "05-手挥琵琶",    6:  "06-左右倒卷肱",
    7:  "07-左揽雀尾",    8:  "08-右揽雀尾",
    9:  "09-单鞭",        10: "10-云手",
    11: "11-单鞭",        12: "12-高探马",
    13: "13-右蹬脚",      14: "14-双峰贯耳",
    15: "15-转身左蹬脚",  16: "16-左下势独立",
    17: "17-右下势独立",  18: "18-左右穿梭",
    19: "19-海底针",      20: "20-闪通臂",
    21: "21-转身搬拦捶",  22: "22-如封似闭",
    23: "23-十字手",      24: "24-收势",
}