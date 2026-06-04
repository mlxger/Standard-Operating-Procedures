# src/pose_estimator.py
# 适配 MediaPipe >= 0.10.x Tasks API

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os
import time
from typing import List, Dict, Optional
import yaml


class HandPoseEstimator:
    """
    MediaPipe Hands Tasks API 版本
    适配 mediapipe >= 0.10.x
    不再使用 mp.solutions，改用 mediapipe.tasks
    """

    # ── 21个关键点骨骼连接定义 ──
    HAND_CONNECTIONS = [
        # 拇指
        (0, 1), (1, 2), (2, 3), (3, 4),
        # 食指
        (0, 5), (5, 6), (6, 7), (7, 8),
        # 中指
        (5, 9), (9, 10), (10, 11), (11, 12),
        # 无名指
        (9, 13), (13, 14), (14, 15), (15, 16),
        # 小指
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    # 指尖关键点 (用于高亮显示)
    FINGERTIP_IDS = [4, 8, 12, 16, 20]

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    MODEL_PATH = "models/mediapipe/hand_landmarker.task"

    def __init__(self, config_path: str = "configs/model_config.yaml"):
        # ── 读取配置 ──
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)['mediapipe']
 
        self.max_num_hands      = cfg.get('max_num_hands', 2)
        self.min_det_conf       = cfg.get('min_detection_confidence', 0.7)
        self.min_track_conf     = cfg.get('min_tracking_confidence', 0.5)
        self.min_presence_conf  = cfg.get('min_presence_confidence', 0.5)

        # ── 确保模型文件存在 ──
        os.makedirs("models", exist_ok=True)
        self._download_model()

        # ── 构建 Tasks API 选项 ──
        base_options = mp_python.BaseOptions(
            model_asset_path=self.MODEL_PATH
        )

        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,   # 视频流追踪模式
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_det_conf,
            min_hand_presence_confidence=self.min_presence_conf,
            min_tracking_confidence=self.min_track_conf,
        )

        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

        # VIDEO 模式需要单调递增时间戳
        self._start_ms = int(time.time() * 1000)

        print("[HandPoseEstimator] ✅ Tasks API 初始化完成")
        print(f"  模型路径  : {self.MODEL_PATH}")
        print(f"  最大手数  : {self.max_num_hands}")
        print(f"  检测置信度: {self.min_det_conf}")

    # ─────────────────────────── 模型下载 ───────────────────────────

    def _download_model(self):
        if os.path.exists(self.MODEL_PATH):
            size_mb = os.path.getsize(self.MODEL_PATH) / 1024 / 1024
            print(f"[HandPoseEstimator] 找到模型文件 ({size_mb:.1f} MB)")
            return

        print("[HandPoseEstimator] 模型文件不存在，开始下载...")
        print(f"  URL: {self.MODEL_URL}")
        try:
            urllib.request.urlretrieve(
                self.MODEL_URL,
                self.MODEL_PATH,
                reporthook=self._progress_hook
            )
            print(f"\n[HandPoseEstimator] ✅ 下载完成: {self.MODEL_PATH}")
        except Exception as e:
            if os.path.exists(self.MODEL_PATH):
                os.remove(self.MODEL_PATH)
            print(f"\n[HandPoseEstimator] ❌ 下载失败: {e}")
            print("  请手动下载并放到 models/hand_landmarker.task")
            print(f"  下载地址: {self.MODEL_URL}")
            raise RuntimeError(f"模型下载失败: {e}")

    @staticmethod
    def _progress_hook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%", end='', flush=True)

    # ─────────────────────────── 核心推理 ───────────────────────────

    def estimate(self, frame: np.ndarray) -> Dict:
        """
        单帧手部关键点检测

        Returns:
            {
              'hands': [
                {
                  'keypoints_norm':  np.ndarray (21, 3),  # x,y,z 归一化
                  'keypoints_pixel': np.ndarray (21, 2),  # 像素坐标
                  'handedness':      str,                 # 'Left'/'Right'
                  'score':           float,               # 置信度
                  'landmarks_raw':   list                 # 原始landmark列表
                }, ...
              ],
              'raw': HandLandmarkerResult
            }
        """
        h, w = frame.shape[:2]

        # BGR → RGB → MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO 模式：时间戳必须严格单调递增 (毫秒)
        timestamp_ms = int(time.time() * 1000) - self._start_ms
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        hands_data = []

        if result.hand_landmarks:
            for idx, hand_lm in enumerate(result.hand_landmarks):

                # 归一化坐标 (21, 3)
                kp_norm = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_lm],
                    dtype=np.float32
                )

                # 像素坐标 (21, 2)
                kp_pixel = np.array(
                    [[int(lm.x * w), int(lm.y * h)] for lm in hand_lm],
                    dtype=np.int32
                )

                # 手型 (Left/Right) + 置信度
                handedness  = "Unknown"
                hand_score  = 0.0
                if result.handedness and idx < len(result.handedness):
                    cat         = result.handedness[idx][0]
                    handedness  = cat.category_name   # "Left" or "Right"
                    hand_score  = cat.score

                hands_data.append({
                    'keypoints_norm':  kp_norm,
                    'keypoints_pixel': kp_pixel,
                    'handedness':      handedness,
                    'score':           hand_score,
                    'landmarks_raw':   hand_lm
                })

        return {'hands': hands_data, 'raw': result}

    # ─────────────────────────── 特征提取 ───────────────────────────

    def get_feature_vector(self, hands_data: List[Dict]) -> np.ndarray:
        """
        提取 126 维特征向量
        格式: [右手21点×3, 左手21点×3] 或全零(无手时)
        """
        features = np.zeros(126, dtype=np.float32)

        # 按手型排序：右手放前半段，左手放后半段
        right_hands = [h for h in hands_data if h['handedness'] == 'Right']
        left_hands  = [h for h in hands_data if h['handedness'] == 'Left']
        ordered     = (right_hands + left_hands)[:2]

        for i, hand in enumerate(ordered):
            flat = hand['keypoints_norm'].flatten()  # (63,)
            features[i * 63: i * 63 + 63] = flat

        return features

    def get_normalized_feature(self, hands_data: List[Dict]) -> np.ndarray:
        """
        提取相对归一化特征 (相对于手腕点，更鲁棒)
        """
        features = np.zeros(126, dtype=np.float32)
        for i, hand in enumerate(hands_data[:2]):
            kp   = hand['keypoints_norm'].copy()   # (21, 3)
            kp  -= kp[0]                            # 以手腕为原点
            # 用手掌长度归一化
            scale = np.linalg.norm(kp[9] - kp[0]) + 1e-6
            kp   /= scale
            features[i * 63: i * 63 + 63] = kp.flatten()
        return features

    # ─────────────────────────── 可视化 ───────────────────────────

    def draw(self, frame: np.ndarray, hand_results: Dict) -> np.ndarray:
        """绘制手部骨骼 + 关键点"""
        vis = frame.copy()

        for hand in hand_results['hands']:
            pts        = hand['keypoints_pixel']   # (21, 2)
            handedness = hand['handedness']
            score      = hand['score']

            # ── 颜色方案 ──
            bone_color = (50, 50, 255)   if handedness == 'Right' else (255, 50, 50)
            tip_color  = (0, 255, 0)
            joint_color= (0, 200, 255)

            # ── 绘制骨骼连线 ──
            for s, e in self.HAND_CONNECTIONS:
                pt1 = tuple(pts[s].tolist())
                pt2 = tuple(pts[e].tolist())
                cv2.line(vis, pt1, pt2, bone_color, 2, cv2.LINE_AA)

            # ── 绘制关键点 ──
            for j, pt in enumerate(pts):
                center = tuple(pt.tolist())
                if j in self.FINGERTIP_IDS:
                    # 指尖：绿色大圆
                    cv2.circle(vis, center, 7, tip_color,   -1, cv2.LINE_AA)
                    cv2.circle(vis, center, 7, (255,255,255), 1, cv2.LINE_AA)
                elif j == 0:
                    # 手腕：橙色大圆
                    cv2.circle(vis, center, 8, (0, 165, 255), -1, cv2.LINE_AA)
                    cv2.circle(vis, center, 8, (255,255,255),  1, cv2.LINE_AA)
                else:
                    cv2.circle(vis, center, 5, joint_color, -1, cv2.LINE_AA)
                    cv2.circle(vis, center, 5, (255,255,255), 1, cv2.LINE_AA)

            # ── 手型标注 ──
            wrist      = tuple(pts[0].tolist())
            label      = f"{handedness[0]} {score:.0%}"
            label_bg   = (30, 30, 200) if handedness == 'Right' else (200, 30, 30)
            (tw, th), _= cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            lx, ly     = wrist[0] - 10, wrist[1] - 20
            cv2.rectangle(vis, (lx, ly - th - 4), (lx + tw + 4, ly + 4), label_bg, -1)
            cv2.putText(vis, label, (lx + 2, ly),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return vis

    def get_hand_bbox(self, hand: Dict,
                      frame_shape: tuple) -> Optional[tuple]:
        """返回手部 bounding box (x1,y1,x2,y2)"""
        px = hand['keypoints_pixel']
        fh, fw = frame_shape[:2]
        x1 = max(0,  int(px[:, 0].min()) - 20)
        y1 = max(0,  int(px[:, 1].min()) - 20)
        x2 = min(fw, int(px[:, 0].max()) + 20)
        y2 = min(fh, int(px[:, 1].max()) + 20)
        return (x1, y1, x2, y2)

    # ─────────────────────────── 资源释放 ───────────────────────────

    def close(self):
        if hasattr(self, 'landmarker') and self.landmarker:
            self.landmarker.close()
            print("[HandPoseEstimator] 资源已释放")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass