"""
TaiChi 24式 视频文件实时推理  ── 稳定性增强版 v3
────────────────────────────────────────────────────────────────
三重稳定保障（缺一不可）:
  ① 多窗口批量推理  : 3个重叠窗口同时forward → 平均概率
                      消除单窗口随机抖动，降低噪声方差
  ② EMA 概率平滑    : alpha=0.15，历史记忆更长
  ③ 严格顺序状态机  : 只允许 current 或 current+1
                      软连续确认 + 最短持续时间双重保护
                      彻底解决 01→05 跳变 & 起势↔收势混淆
────────────────────────────────────────────────────────────────
"""
import os
import time
import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import deque
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

from config import *
from model import TaiChiTransformer


# ══════════════════════════════════════════════════════════════════
#  GUI 可用性检测
# ══════════════════════════════════════════════════════════════════
def _check_headless() -> bool:
    try:
        cv2.imshow("__probe__", np.zeros((4, 4, 3), np.uint8))
        cv2.waitKey(1)
        cv2.destroyWindow("__probe__")
        return False
    except cv2.error:
        return True

_HEADLESS = _check_headless()
if _HEADLESS:
    print("⚠  无头模式：结果将写入 _result.mp4\n")


# ── 骨骼连接 ──────────────────────────────────────────────────────
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]
_FACE_IDX  = {0,1,2,3,4,5,6,7,8,9,10}
_LEFT_IDX  = {11,13,15,17,19,21,23,25,27,29,31}
_RIGHT_IDX = {12,14,16,18,20,22,24,26,28,30,32}

def _conn_color(a: int, b: int) -> Tuple[int,int,int]:
    if a in _FACE_IDX and b in _FACE_IDX: return (180,180,180)
    if a in _LEFT_IDX  or b in _LEFT_IDX:  return (0,230,128)
    if a in _RIGHT_IDX or b in _RIGHT_IDX: return (0,160,255)
    return (200,200,0)


# ══════════════════════════════════════════════════════════════════
#  严格顺序状态机
# ══════════════════════════════════════════════════════════════════
class TaiChiStateMachine:
    """
    太极24式严格单向状态机
    ───────────────────────────────────────────────────────────────
    设计原则:
      • 合法状态只有两个: 停留(state) 或 前进(state+1)
      • 其他所有动作的概率被完全遮蔽 → 物理消除跳变和混淆
      • 前进必须满足两个独立条件:
          ① hold_cnt ≥ min_hold  （最短持续时间保护）
          ② cand_cnt ≥ confirm_n （软连续确认，容忍偶发噪声）
      • cand_cnt 使用软衰减而非硬重置，对过渡期更友好

    参数说明 (以 src_fps=25, infer_every=STRIDE=8 为例):
      每秒推理次数 = 25/8 ≈ 3.1次
      min_hold=10  → 每式至少驻留 ~3.2秒
      confirm_n=5  → 净确认需要累积到5次（允许中间出现1-2次噪声）
    ───────────────────────────────────────────────────────────────
    """
    def __init__(self,
                 n_classes:  int = NUM_CLASSES,
                 min_hold:   int = 10,
                 confirm_n:  int = 5,
                 init_state: int = 0):
        self.n         = n_classes
        self.min_hold  = min_hold
        self.confirm_n = confirm_n
        self.state     = init_state
        self.hold_cnt  = 0    # 当前状态已推理次数
        self.cand_cnt  = 0    # 净切换证据（+1 next胜, -1 current胜）

    def reset(self, state: int = 0) -> None:
        self.state    = state
        self.hold_cnt = 0
        self.cand_cnt = 0

    def update(self, probs: np.ndarray) -> Tuple[int, float]:
        """
        输入: probs (NUM_CLASSES,) — EMA 平滑后的概率向量
        输出: (confirmed_state 0-indexed,  masked_conf)
        """
        next_s = (self.state + 1) % self.n

        # ① 遮蔽所有非法动作，只保留 current 和 next
        masked = np.zeros(self.n, dtype=np.float32)
        masked[self.state] = probs[self.state]
        masked[next_s]     = probs[next_s]
        denom = masked.sum()
        if denom > 1e-9:
            masked /= denom
        else:
            masked[self.state] = 1.0   # 兜底：保持当前

        # ② 累积持续计数
        self.hold_cnt += 1

        # ③ 软连续确认（next胜 → +1；current胜 → -1，最低为0）
        #    与硬重置相比，允许在动作过渡期出现少量噪声
        if (self.hold_cnt >= self.min_hold
                and masked[next_s] > masked[self.state]):
            self.cand_cnt += 1
        else:
            self.cand_cnt = max(0, self.cand_cnt - 1)

        # ④ 达到确认阈值 → 切换到下一动作
        if self.cand_cnt >= self.confirm_n:
            self.state    = next_s
            self.hold_cnt = 0
            self.cand_cnt = 0

        return self.state, float(masked[self.state])

    @property
    def transition_progress(self) -> float:
        """切换进度 [0,1]，用于 UI 进度条"""
        if self.hold_cnt < self.min_hold:
            return 0.0
        return min(self.cand_cnt / max(self.confirm_n, 1), 1.0)

    @property
    def is_unlocked(self) -> bool:
        """最短持续时间是否已满足（可以开始积累确认）"""
        return self.hold_cnt >= self.min_hold


# ══════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════
def put_chinese(img_bgr: np.ndarray,
                text: str,
                pos: Tuple[int,int],
                font_size: int = 24,
                color: Tuple[int,int,int] = (255,255,255)) -> np.ndarray:
    """借助 PIL 在 OpenCV BGR 图像上渲染中文"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil_img)
    font    = ImageFont.load_default()
    for fp in ["simhei.ttf",
               "C:/Windows/Fonts/simhei.ttf",
               "C:/Windows/Fonts/msyh.ttc",
               "C:/Windows/Fonts/simsun.ttc"]:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except (IOError, OSError):
            continue
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def normalize_sequence(kps_seq: np.ndarray) -> np.ndarray:
    """(T, 99) → (T, 198)  髋部中心化 + 躯干缩放 + 速度特征"""
    T      = len(kps_seq)
    kps_3d = kps_seq.reshape(T, 33, 3)
    hip    = (kps_3d[:,23] + kps_3d[:,24]) / 2.0
    sho    = (kps_3d[:,11] + kps_3d[:,12]) / 2.0
    scale  = np.linalg.norm(sho - hip, axis=1, keepdims=True).clip(1e-6)
    pos    = ((kps_3d - hip[:,None,:]) / scale[:,None,:]).reshape(T, FEATURE_DIM_POS)
    vel    = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    vel[0]  = vel[1]
    return np.concatenate([pos, vel], axis=-1)   # (T, 198)


# ══════════════════════════════════════════════════════════════════
#  主推理类
# ══════════════════════════════════════════════════════════════════
class TaiChiPredictor:
    """
    完整推理流程（每帧）:
      MediaPipe 姿态检测
          ↓
      关键点存入长缓冲区 (long_buf, 容量=96帧)
          ↓ 每 STRIDE=8 帧触发一次
      ① 多窗口批量推理: 从 long_buf 取3个重叠窗口 → 1次batch forward → 均值概率
          ↓
      ② EMA 平滑 (alpha=0.15)
          ↓
      ③ 顺序状态机 (只允许 current / current+1，软确认)
          ↓
      UI 渲染
    ─────────────────────────────────────────────────────────────
    多窗口布局示意 (NUM_WINS=3, WIN_STRIDE=16, SEQ_LEN=64):
      long_buf: |────────── 96 帧 ──────────|
      win_2:    |── 64 ──|                    (最旧，t-95~t-32)
      win_1:         |── 64 ──|              (中间，t-79~t-16)
      win_0:               |── 64 ──|        (最新，t-63~t)
    """

    _NUM_WINS   = 3    # 并行推理窗口数
    _WIN_STRIDE = 16   # 相邻窗口偏移帧数

    def __init__(self,
                 model_path:      str = './checkpoints/taichi_transformer_best.pth',
                 min_hold_infers: int = 10,
                 confirm_infers:  int = 5):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"推理设备: {self.device}")

        # ── 加载模型 ───────────────────────────────────────────────
        self.model = TaiChiTransformer().to(self.device)
        try:
            ckpt = torch.load(model_path, map_location=self.device,
                              weights_only=True)
        except Exception as e:
            print(f"  weights_only=True 失败 ({e})，降级")
            ckpt = torch.load(model_path, map_location=self.device,
                              weights_only=False)
        self.model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        self.model.eval()
        print(f"✓ 模型: {model_path}")

        # ── MediaPipe PoseLandmarker (VIDEO 模式) ──────────────────
        base_opts = mp_python.BaseOptions(
            model_asset_path=MEDIAPIPE_MODEL_PATH
        )
        lm_opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self.detector = mp_vision.PoseLandmarker.create_from_options(lm_opts)
        print("✓ MediaPipe 初始化完成\n")

        # ── 长缓冲区（为多窗口保存足够历史） ──────────────────────
        # 长度 = SEQ_LEN + (NUM_WINS-1)*WIN_STRIDE = 64 + 2*16 = 96
        self._buf_len = SEQ_LEN + (self._NUM_WINS - 1) * self._WIN_STRIDE
        self.long_buf = deque(maxlen=self._buf_len)
        self.last_kps = np.zeros(FEATURE_DIM_POS, dtype=np.float32)

        # ── EMA 平滑（alpha 越低越稳定，响应越慢）─────────────────
        self.ema_probs = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
        self.ema_alpha = 0.15

        # ── 顺序状态机 ─────────────────────────────────────────────
        self.fsm = TaiChiStateMachine(
            min_hold  = min_hold_infers,
            confirm_n = confirm_infers,
        )

        # ── 推理频率（只赋值一次，与训练 stride 对齐）─────────────
        self.infer_every = STRIDE   # = 8

        # ── 置信度显示阈值 ─────────────────────────────────────────
        self.conf_thresh = 0.40   # 归一化后概率，较易满足

        # ── 展示状态 ───────────────────────────────────────────────
        self.cur_action: Optional[int] = None
        self.cur_conf:   float         = 0.0
        self.top3_ema:   list          = []   # 未经状态机约束的 EMA Top3

        # ── FPS 统计 ───────────────────────────────────────────────
        self.fps_buf = deque(maxlen=30)
        self.prev_t  = time.time()

    # ── 关键点提取 ─────────────────────────────────────────────────
    def _extract_kps(self, result) -> Optional[np.ndarray]:
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            kps = []
            for lm in result.pose_landmarks[0]:
                kps.extend([lm.x, lm.y, lm.z])
            return np.array(kps, dtype=np.float32)
        return None

    # ── ① 多窗口批量推理 ──────────────────────────────────────────
    @torch.no_grad()
    def _infer_multi_window(self) -> np.ndarray:
        """
        从 long_buf 采样 _NUM_WINS 个重叠窗口，
        拼成一个 batch 做一次 forward，返回平均概率 (NUM_CLASSES,)。

        批量 forward 比 N 次单独 forward 更高效，GPU 利用率更好。
        平均操作将单窗口方差降低到 1/N，显著减少抖动。
        """
        history = np.array(list(self.long_buf))   # (buf_len, 99)
        T       = len(history)

        windows = []
        for i in range(self._NUM_WINS):
            offset = i * self._WIN_STRIDE   # 0, 16, 32
            end    = T - offset
            start  = end - SEQ_LEN
            if start < 0:
                break
            windows.append(history[start:end])   # (64, 99)

        if not windows:
            return np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES

        # 归一化后拼成 batch → 单次 forward
        batch  = np.stack([normalize_sequence(w) for w in windows])  # (N, 64, 198)
        tensor = torch.FloatTensor(batch).to(self.device)
        logits = self.model(tensor)                                   # (N, 24)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()         # (N, 24)

        return probs.mean(axis=0)   # (24,)  均值概率，方差 ≈ 单窗口的 1/3

    # ── ② EMA 平滑 ────────────────────────────────────────────────
    def _ema_update(self, raw: np.ndarray) -> np.ndarray:
        self.ema_probs = (      self.ema_alpha  * raw
                         + (1 - self.ema_alpha) * self.ema_probs)
        return self.ema_probs

    # ── 重置（键盘 R） ─────────────────────────────────────────────
    def reset(self) -> None:
        self.long_buf.clear()
        self.last_kps  = np.zeros(FEATURE_DIM_POS, dtype=np.float32)
        self.ema_probs = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
        self.fsm.reset(0)
        self.cur_action = None
        self.cur_conf   = 0.0
        self.top3_ema   = []
        print("\n↺ 已重置到动作 01-起势")

    # ── 骨骼绘制 ───────────────────────────────────────────────────
    def _draw_skeleton(self, frame_bgr: np.ndarray, result) -> np.ndarray:
        if not result.pose_landmarks:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        for lm_list in result.pose_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
            n   = len(pts)
            for a, b in POSE_CONNECTIONS:
                if a < n and b < n:
                    cv2.line(frame_bgr, pts[a], pts[b],
                             _conn_color(a, b), 2, cv2.LINE_AA)
            for i, pt in enumerate(pts):
                clr = ((180,180,180) if i in _FACE_IDX  else
                       (0,230,128)   if i in _LEFT_IDX  else
                       (0,160,255)   if i in _RIGHT_IDX else
                       (200,200,0))
                cv2.circle(frame_bgr, pt, 5, clr,        -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, pt, 5, (20,20,20),  1, cv2.LINE_AA)
        return frame_bgr

    # ── UI 绘制 ────────────────────────────────────────────────────
    def _draw_ui(self, frame: np.ndarray, fps: float,
                 cur_frame: int, total_frames: int) -> np.ndarray:
        h, w = frame.shape[:2]
        ov   = frame.copy()
        cv2.rectangle(ov, (0,0),     (415, 310), (12,12,12), -1)
        cv2.rectangle(ov, (w-315,0), (w,   260), (12,12,12), -1)
        cv2.rectangle(ov, (0,h-52),  (w,   h),   (12,12,12), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

        # ── FPS + 帧计数 ───────────────────────────────────────────
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (60,240,60), 2)
        cv2.putText(frame, f"Frame {cur_frame:5d}/{total_frames}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170,170,170), 1)

        # ── 主预测区 ───────────────────────────────────────────────
        if self.cur_action is not None and self.cur_conf >= self.conf_thresh:
            aid  = self.cur_action + 1
            name = ACTION_NAMES.get(aid, f"Action_{aid}")

            # 动作编号 + 中文名
            cv2.putText(frame, f"Action {aid:02d}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,240,240), 2)
            frame = put_chinese(frame, name, (10, 108),
                                font_size=32, color=(255,255,255))

            # 当前动作置信度条
            bmax = 340
            bw   = int(bmax * self.cur_conf)
            bclr = ((0,215,0)   if self.cur_conf > 0.75 else
                    (0,160,255) if self.cur_conf > 0.50 else
                    (0,70,255))
            cv2.rectangle(frame, (10,168), (10+bmax,186), (55,55,55), -1)
            cv2.rectangle(frame, (10,168), (10+bw,  186), bclr,       -1)
            cv2.putText(frame, f"{self.cur_conf*100:.1f}%",
                        (bmax+16, 184), cv2.FONT_HERSHEY_SIMPLEX,
                        0.68, bclr, 2)

            # 下一式 + 切换进度条
            next_aid  = (self.cur_action + 1) % NUM_CLASSES + 1
            next_name = ACTION_NAMES.get(next_aid, f"Act{next_aid}")
            prog      = self.fsm.transition_progress        # [0, 1]
            prog_w    = int(bmax * prog)
            pclr      = (0,200,255) if prog < 0.6 else (0,215,0)
            frame = put_chinese(frame,
                                f"→ {next_name}",
                                (10, 196), font_size=20, color=(150,150,150))
            cv2.rectangle(frame, (10,222), (10+bmax,236), (40,40,40), -1)
            cv2.rectangle(frame, (10,222), (10+prog_w,236), pclr,     -1)
            cv2.putText(frame,
                        f"确认进度 {int(prog*100)}%"
                        f"  ({self.fsm.cand_cnt}/{self.fsm.confirm_n})",
                        (bmax+16, 235),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, pclr, 1)

            # 状态机锁定状态
            if self.fsm.is_unlocked:
                lock_txt = f"✓ 可切换  驻留 {self.fsm.hold_cnt}/{self.fsm.min_hold}"
                lock_clr = (0, 215, 0)
            else:
                lock_txt = f"🔒 锁定中  驻留 {self.fsm.hold_cnt}/{self.fsm.min_hold}"
                lock_clr = (100, 100, 100)
            cv2.putText(frame, lock_txt, (10, 265),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, lock_clr, 1)

        else:
            ratio  = len(self.long_buf) / self._buf_len
            status = (f"预热中... {int(ratio*100)}%"
                      if ratio < 1.0 else "推理中...")
            frame  = put_chinese(frame, status, (10, 90),
                                 font_size=28, color=(150,150,150))

        # ── 右上 Top-3（来自EMA，未经状态机约束）─────────────────
        #    亮色 = 状态机允许的动作；暗色 = 当前被屏蔽
        if self.top3_ema:
            frame = put_chinese(frame, "Top-3 EMA",
                                (w-305, 8), font_size=20, color=(190,190,190))
            fsm_cur  = self.fsm.state
            fsm_next = (self.fsm.state + 1) % NUM_CLASSES
            for rank, (cls, prob) in enumerate(self.top3_ema):
                y    = 36 + rank * 70
                name = ACTION_NAMES.get(cls + 1, f"Act{cls+1}")
                if cls == fsm_cur:
                    clr = (0, 240, 240)    # 青色：当前确认动作
                elif cls == fsm_next:
                    clr = (0, 215, 100)    # 绿色：待切换动作
                else:
                    clr = (90, 90, 90)     # 灰色：被状态机屏蔽
                frame = put_chinese(frame, f"#{rank+1} {name}",
                                    (w-305, y), font_size=20, color=clr)
                mini_w = int(240 * prob)
                cv2.rectangle(frame, (w-305,y+28),(w-305+240,y+40),
                              (50,50,50), -1)
                cv2.rectangle(frame, (w-305,y+28),(w-305+mini_w,y+40),
                              clr, -1)
                cv2.putText(frame, f"{prob*100:.1f}%",
                            (w-58, y+40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, clr, 1)

        # ── 底部：缓冲进度条 + 快捷键 ─────────────────────────────
        buf_fill = int((w-20) * min(len(self.long_buf) / self._buf_len, 1.0))
        cv2.rectangle(frame, (10,h-40), (w-10,h-16), (50,50,50), -1)
        cv2.rectangle(frame, (10,h-40), (10+buf_fill,h-16), (60,185,60), -1)
        hint = (f"Buf {len(self.long_buf):2d}/{self._buf_len} "
                f"│ FSM: {self.fsm.state+1:02d}→{(self.fsm.state+1)%NUM_CLASSES+1:02d} "
                f"│ {'输出中...' if _HEADLESS else '[Q]退出 [Space]暂停 [S]截图 [R]重置序列'}")
        cv2.putText(frame, hint, (15, h-19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (205,205,205), 1)
        return frame

    # ── 主循环 ─────────────────────────────────────────────────────
    def run(self, video_path: str) -> None:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"视频不存在: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 根据视频FPS打印参数的时间含义，方便调参
        infers_ps = fps_src / self.infer_every
        print(f"视频: {os.path.basename(video_path)}  {W}×{H}  "
              f"{fps_src:.1f}fps  {total_frames}帧")
        print(f"每秒推理次数: {infers_ps:.2f}")
        print(f"min_hold={self.fsm.min_hold} "
              f"→ 每式最短驻留 {self.fsm.min_hold/infers_ps:.1f}s")
        print(f"confirm_n={self.fsm.confirm_n} "
              f"→ 切换需净积累 {self.fsm.confirm_n} 次确认\n")

        writer   = None
        out_path = ""
        if _HEADLESS:
            out_path = os.path.splitext(video_path)[0] + "_result.mp4"
            writer   = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_src, (W, H)
            )
            print(f"无头模式 → {out_path}\n")

        frame_idx = 0
        paused    = False
        display   = np.zeros((H, W, 3), dtype=np.uint8)

        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("\n视频播放完毕。")
                        break

                    # ── MediaPipe 姿态检测 ──────────────────────────
                    ts_ms  = int(frame_idx * 1000.0 / fps_src)
                    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=rgb)
                    result = self.detector.detect_for_video(mp_img, ts_ms)

                    # ── 关键点入长缓冲（前向填充） ──────────────────
                    kps = self._extract_kps(result)
                    if kps is not None:
                        self.last_kps = kps
                    self.long_buf.append(self.last_kps.copy())

                    # ── 三重稳定推理（缓冲区满后，每 STRIDE 帧一次）
                    if (len(self.long_buf) == self._buf_len
                            and frame_idx % self.infer_every == 0):

                        raw_probs = self._infer_multi_window()    # ① 多窗口均值
                        ema_p     = self._ema_update(raw_probs)   # ② EMA 平滑
                        act, conf = self.fsm.update(ema_p)        # ③ 顺序状态机

                        self.cur_action = act
                        self.cur_conf   = conf

                        # Top-3 取自 EMA（未经状态机遮蔽），用于调试观察
                        idx           = ema_p.argsort()[::-1][:3]
                        self.top3_ema = [(int(i), float(ema_p[i])) for i in idx]

                    # ── FPS 统计 ────────────────────────────────────
                    now = time.time()
                    self.fps_buf.append(1.0 / max(now - self.prev_t, 1e-6))
                    self.prev_t = now
                    avg_fps     = float(np.mean(self.fps_buf))

                    # ── 绘制 ────────────────────────────────────────
                    display = frame.copy()
                    display = self._draw_skeleton(display, result)
                    display = self._draw_ui(display, avg_fps,
                                           frame_idx, total_frames)

                    if _HEADLESS:
                        writer.write(display)
                        if frame_idx % 30 == 0:
                            s = "预热中"
                            if self.cur_action is not None:
                                aid = self.cur_action + 1
                                nm  = ACTION_NAMES.get(aid, f"Act{aid}")
                                s   = f"{nm} ({self.cur_conf*100:.1f}%)"
                            print(f"\r  帧{frame_idx:5d}/{total_frames}"
                                  f"  FPS:{avg_fps:5.1f}  {s}",
                                  end="", flush=True)
                    else:
                        cv2.imshow(
                            "TaiChi 24 — Stable Recognition", display
                        )

                    frame_idx += 1

                # ── 键盘事件（仅 GUI 模式）─────────────────────────
                if not _HEADLESS:
                    key = cv2.waitKey(1) & 0xFF
                    if   key == ord('q'):
                        print("退出。"); break
                    elif key == ord(' '):
                        paused = not paused
                        print("⏸ 暂停" if paused else "▶ 继续")
                    elif key == ord('s'):
                        fn = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                        cv2.imwrite(fn, display)
                        print(f"截图: {fn}")
                    elif key == ord('r'):
                        self.reset()

        finally:
            cap.release()
            if _HEADLESS and writer:
                writer.release()
                print(f"\n✓ 已保存: {out_path}")
            else:
                cv2.destroyAllWindows()
            self.detector.close()
            print("资源已释放。")


# ── 入口 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="TaiChi 24式视频推理（稳定版）")
    p.add_argument('--model',
                   default='./checkpoints/taichi_transformer_best.pth')
    p.add_argument('--video',   required=True)
    p.add_argument('--min-hold', type=int, default=10,
                   help='每式最短驻留推理次数 (默认10 ≈ 3.2s @ 25fps/STRIDE=8)')
    p.add_argument('--confirm',  type=int, default=5,
                   help='切换所需净确认次数 (默认5 ≈ 1.6s 净积累)')
    args = p.parse_args()

    predictor = TaiChiPredictor(
        model_path      = args.model,
        min_hold_infers = args.min_hold,
        confirm_infers  = args.confirm,
    )
    predictor.run(video_path=args.video)