"""
从 taichi_frames 各子文件夹提取 MediaPipe Pose 关键点
每个文件夹固定 103 帧
使用 MediaPipe Tasks API (0.10.x) — PoseLandmarker IMAGE 模式
输出 .npy 文件: shape = (103, 99)
"""
import os
import re
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

from config import (DATA_ROOT, KEYPOINTS_DIR, FEATURE_DIM,
                    MEDIAPIPE_MODEL_PATH, FRAMES_PER_SAMPLE)


# ── 工具函数 ───────────────────────────────────────────────────────
def get_all_folders(data_root: str):
    """解析文件夹名称，返回 (folder_name, action_id, sample_id) 列表"""
    pattern = re.compile(r'^(\d{2})_(\d{2})$')
    folders = []
    for name in sorted(os.listdir(data_root)):
        match = pattern.match(name)
        if match and os.path.isdir(os.path.join(data_root, name)):
            folders.append((name, int(match.group(1)), int(match.group(2))))
    return folders


def build_image_detector() -> mp_vision.PoseLandmarker:
    """
    创建 Tasks API PoseLandmarker（IMAGE 模式）
    IMAGE 模式：逐帧独立检测，无时序追踪，适合批量离线处理
    """
    base_options = mp_python.BaseOptions(
        model_asset_path=MEDIAPIPE_MODEL_PATH
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_keypoints_from_result(result) -> np.ndarray:
    """
    从 Tasks API PoseLandmarkerResult 提取关键点向量 (99,)
    result.pose_landmarks: List[List[NormalizedLandmark]]
    若未检测到则返回全零向量
    """
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        kps = []
        # result.pose_landmarks[0] → 第一个人的 33 个关键点
        for lm in result.pose_landmarks[0]:
            kps.extend([lm.x, lm.y, lm.z])
        return np.array(kps, dtype=np.float32)
    return None  # 未检测到


def extract_folder(frames_dir: str,
                   detector: mp_vision.PoseLandmarker) -> np.ndarray:
    """
    处理单个文件夹中全部帧（固定103帧），返回关键点时序矩阵
    返回 shape: (103, 99)
    未检测到关键点的帧：复用上一帧（前向填充）
    """
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])

    keypoints_seq = []
    last_valid    = np.zeros(FEATURE_DIM, dtype=np.float32)

    for fname in frame_files:
        img_path  = os.path.join(frames_dir, fname)
        bgr_image = cv2.imread(img_path)

        if bgr_image is None:
            # 图片读取失败：前向填充
            keypoints_seq.append(last_valid.copy())
            continue

        # Tasks API 要求 RGB uint8 numpy array
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=rgb_image)
        result    = detector.detect(mp_image)

        kps = extract_keypoints_from_result(result)
        if kps is not None:
            last_valid = kps  # 更新前向填充缓存

        keypoints_seq.append(last_valid.copy())

    return np.array(keypoints_seq, dtype=np.float32)


# ── 主提取函数 ─────────────────────────────────────────────────────
def extract_all(data_root: str = DATA_ROOT,
                save_dir: str  = KEYPOINTS_DIR):
    os.makedirs(save_dir, exist_ok=True)

    folders = get_all_folders(data_root)
    print(f"共发现 {len(folders)} 个样本文件夹 "
          f"(动作01~24, 每动作最多53个样本)\n")

    stats   = {}   # action_id → [frame_count, ...]
    missing = []   # 未成功提取的文件夹

    # ── 创建 detector（可复用，整个提取过程只创建一次） ───────────
    detector = build_image_detector()

    try:
        for folder_name, action_id, sample_id in tqdm(
                folders, desc="提取关键点", ncols=90):

            save_path = os.path.join(save_dir, f"{folder_name}.npy")
            if os.path.exists(save_path):
                continue  # 断点续传：已存在则跳过

            frames_dir = os.path.join(data_root, folder_name)
            kps        = extract_folder(frames_dir, detector)

            if len(kps) == 0:
                print(f"\n  ⚠ {folder_name}: 无有效帧，跳过")
                missing.append(folder_name)
                continue

            # 验证帧数（期望固定103帧）
            if len(kps) != FRAMES_PER_SAMPLE:
                print(f"\n  ⚠ {folder_name}: 实际帧数={len(kps)} "
                      f"≠ 预期 {FRAMES_PER_SAMPLE}")

            np.save(save_path, kps)
            stats.setdefault(action_id, []).append(len(kps))

    finally:
        # 务必显式关闭 detector，释放原生资源
        detector.close()

    # ── 打印统计摘要 ────────────────────────────────────────────
    print("\n══════════════ 提取完成 ══════════════")
    total = sum(len(v) for v in stats.values())
    print(f"  成功提取: {total} 个样本")
    for action_id in sorted(stats.keys()):
        counts = stats[action_id]
        print(f"  动作 {action_id:02d}: {len(counts):2d} 个样本  "
              f"帧数 avg={np.mean(counts):.0f}  "
              f"min={min(counts)}  max={max(counts)}")
    if missing:
        print(f"\n  ✗ 失败样本 ({len(missing)}个): {missing}")
    print("═════════════════════════════════════")


if __name__ == "__main__":
    extract_all()