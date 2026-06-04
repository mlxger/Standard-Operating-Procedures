import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# ============================================================
# 手部 21 个关键点的连接关系（用于手动绘制）
# ============================================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 大拇指
    (0, 5), (5, 6), (6, 7), (7, 8),           # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),      # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),    # 无名指
    (13, 17), (17, 18), (18, 19), (19, 20),   # 小指
    (0, 17)                                    # 手掌外侧边缘
]

# 关键点索引常量（对应旧版 mp_hands.HandLandmark）
INDEX_FINGER_TIP = 8


def draw_hand_landmarks(image, hand_landmarks, handedness_label=None):
    """
    在图像上绘制手部关键点和骨骼连接线。
    hand_landmarks: List[NormalizedLandmark]，坐标均为 0~1 归一化值
    """
    h, w = image.shape[:2]

    # 绘制骨骼连接线（绿色）
    for start_idx, end_idx in HAND_CONNECTIONS:
        start_lm = hand_landmarks[start_idx]
        end_lm   = hand_landmarks[end_idx]
        pt1 = (int(start_lm.x * w), int(start_lm.y * h))
        pt2 = (int(end_lm.x * w),   int(end_lm.y * h))
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

    # 绘制关键点（红色实心圆 + 白色外圈）
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 6, (0, 0, 255), -1)
        cv2.circle(image, (cx, cy), 6, (255, 255, 255), 1)

    # 在腕部（关键点0）显示左右手标签
    if handedness_label:
        wrist = hand_landmarks[0]
        wx, wy = int(wrist.x * w), int(wrist.y * h)
        cv2.putText(image, handedness_label, (wx - 30, wy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)


# ============================================================
# 模型路径（将下载好的文件放到此路径）
# ============================================================
MODEL_PATH = './models/mediapipe/hand_landmarker.task'


# ============================================================
# 静态图片检测
# ============================================================
IMAGE_FILES = []  # 例如：['test1.jpg', 'test2.png']

static_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

with vision.HandLandmarker.create_from_options(static_options) as landmarker:
    for idx, file in enumerate(IMAGE_FILES):
        # 沿 y 轴翻转（与旧版行为一致）
        bgr_image = cv2.flip(cv2.imread(file), 1)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # 转为 MediaPipe Image 对象
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        results = landmarker.detect(mp_image)

        print('Handedness:', results.handedness)
        if not results.hand_landmarks:
            continue

        image_height, image_width = bgr_image.shape[:2]
        annotated_image = bgr_image.copy()

        for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
            # 获取左右手标签
            label = ''
            if results.handedness and len(results.handedness) > hand_idx:
                label = results.handedness[hand_idx][0].display_name
                print(f'  Hand {hand_idx} - {label}')

            # 打印食指指尖坐标
            tip = hand_landmarks[INDEX_FINGER_TIP]
            print(f'  Index finger tip: '
                  f'({tip.x * image_width:.1f}, {tip.y * image_height:.1f})')

            draw_hand_landmarks(annotated_image, hand_landmarks, label)

        # 翻转回去后保存
        cv2.imwrite(f'/tmp/annotated_image{idx}.png', cv2.flip(annotated_image, 1))
        print(f'已保存: /tmp/annotated_image{idx}.png')


# ============================================================
# 摄像头实时检测
# ============================================================
cap = cv2.VideoCapture(0)

# 摄像头使用 VIDEO 模式（逐帧同步处理，比 LIVE_STREAM 更简单）
# VIDEO 模式要求：每帧的 timestamp_ms 必须单调递增
video_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

with vision.HandLandmarker.create_from_options(video_options) as landmarker:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 获取单调递增时间戳（毫秒）—— VIDEO 模式强制要求
        timestamp_ms = int(time.time() * 1000)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # 视频模式用 detect_for_video，而非 detect
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
                label = ''
                if results.handedness and len(results.handedness) > hand_idx:
                    label = results.handedness[hand_idx][0].display_name
                draw_hand_landmarks(image, hand_landmarks, label)

        # 水平翻转为"镜像自拍"视角后显示
        cv2.imshow('MediaPipe Hands (New API)', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:  # 按 ESC 退出
            break

cap.release()
cv2.destroyAllWindows()