"""
运行方式: python tools/data_collector.py
快捷键:
  0-9: 标记当前动作类别
  s  : 保存当前标签的序列
  q  : 退出
"""
import cv2
import sys
import os
import numpy as np
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pose_estimator import HandPoseEstimator

SAVE_DIR = Path("data/keypoints")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {
    '0': '放挡圈', '1': '螺钉1',   '2': '放置转子',
    '3': '放螺母', '4': '点胶',    '5': '装转块',
    '6': '空闲',   '7': '拿取零件','8': '拧螺钉', '9': '检查'
}

def main():
    estimator = HandPoseEstimator()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    buffer   = []
    cur_label = None
    SEQ_LEN  = 30  # 1秒30帧
    
    print("=== 数据采集工具 ===")
    print("按 0-9 选择动作类别，按住期间自动录制，松开后保存")
    print("按 q 退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        pose = estimator.estimate(frame)
        vis  = estimator.draw(frame, pose)
        feat = estimator.get_feature_vector(pose['hands'])
        
        # UI信息
        cv2.putText(vis, f"当前动作: {LABELS.get(cur_label, '未选择')}",
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, f"缓冲: {len(buffer)}/{SEQ_LEN}",
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 显示所有标签
        for i, (k, v) in enumerate(LABELS.items()):
            cv2.putText(vis, f"[{k}] {v}",
                       (10, 120 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        
        if cur_label is not None:
            buffer.append(feat)
            if len(buffer) == SEQ_LEN:
                _save_sample(buffer, int(cur_label), LABELS[cur_label])
                buffer = []
        
        cv2.imshow("Data Collector", vis)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif chr(key) in LABELS:
            cur_label = chr(key)
            buffer    = []
        elif key == 32:  # 空格 - 停止录制
            cur_label = None
            buffer    = []
    
    cap.release()
    cv2.destroyAllWindows()

def _save_sample(buffer, label_id: int, label_name: str):
    seq = np.array(buffer)  # (30, 126)
    files = list((SAVE_DIR / str(label_id)).glob("*.npy"))
    count = len(files)
    
    save_path = SAVE_DIR / str(label_id)
    save_path.mkdir(exist_ok=True)
    np.save(save_path / f"{count:05d}.npy", seq)
    print(f"[保存] 类别: {label_name}({label_id}) | 样本#{count} | shape={seq.shape}")

if __name__ == "__main__":
    main()