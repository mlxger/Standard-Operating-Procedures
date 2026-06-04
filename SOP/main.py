import sys
import cv2
import numpy as np
from typing import Dict, Optional

from src.detector         import ObjectDetector
from src.pose_estimator   import HandPoseEstimator
from src.tracker          import ObjectTracker
from src.action_recognizer import ActionRecognizer
from src.sop_engine       import SOPEngine
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class SOPSystem:
    """AI SOP行为分析主系统"""
    
    def __init__(self, camera_id: int = 0):
        print("=" * 50)
        print("  SOP 组装行为分析系统 启动中...")
        print("=" * 50)
        
        self.detector   = ObjectDetector()
        self.pose       = HandPoseEstimator()
        self.tracker    = ObjectTracker()
        self.recognizer = ActionRecognizer()
        self.engine     = SOPEngine()
        
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self.cap.set(cv2.CAP_PROP_FPS,            30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,      1)  # 减少延迟
        
        assert self.cap.isOpened(), "摄像头打开失败!"
        print("  ✅ 摄像头初始化成功")
        print("=" * 50)
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        # ① 目标检测
        det_results = self.detector.detect(frame)
        
        # ② 手部姿态估计
        pose_results = self.pose.estimate(frame)
        
        # ③ 目标追踪
        tracked = self.tracker.update(det_results['detections'], frame)
        
        # ④ 提取特征向量
        features = self.pose.get_feature_vector(pose_results['hands'])
        
        # ⑤ 动作识别
        action = self.recognizer.update(features)
        
        # ⑥ SOP状态更新
        sop_state = self.engine.update(action)
        
        # ⑦ 可视化合成
        vis = self._render(frame, pose_results, tracked, action, sop_state)
        
        return {'frame': vis, 'sop_state': sop_state, 'action': action}
    
    def _render(self, frame, pose_results, tracked,
                action, sop_state) -> np.ndarray:
        # 手部骨骼
        vis = self.pose.draw(frame, pose_results)
        # 追踪框
        vis = self.tracker.draw_tracks(vis, tracked)
        
        # 当前动作
        if action:
            color = (0, 255, 0) if action['confidence'] > 0.75 else (0, 165, 255)
            cv2.putText(
                vis,
                f"动作: {action['action_name']}  ({action['confidence']:.0%})",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
            )
        
        # 当前步骤
        idx   = sop_state['current_idx']
        steps = sop_state['steps']
        if idx < len(steps):
            step = steps[idx]
            status_color = {
                'PASS':    (0, 200, 0),
                'FAIL':    (0, 0, 220),
                'Running': (0, 165, 255),
                'Idle':    (150, 150, 150),
            }.get(step.status.value, (150, 150, 150))
            
            cv2.putText(
                vis,
                f"步骤 {step.step_id}: {step.name}  [{step.status.value}]",
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2
            )
        
        # 底部统计
        info = (f"良率:{sop_state['yield_rate']:.1f}%  "
                f"良品:{sop_state['good_products']}  "
                f"不良:{sop_state['bad_products']}")
        cv2.putText(vis, info, (20, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis


def run_with_ui(camera_id: int = 0):
    """PyQt5 图形界面模式"""
    from PyQt5.QtWidgets import QApplication
    from dashboard import SOPDashboard
    
    app    = QApplication(sys.argv)
    system = SOPSystem(camera_id)
    win    = SOPDashboard(system)
    win.show()
    sys.exit(app.exec_())


def run_headless(camera_id: int = 0):
    """纯OpenCV 无UI模式 (低资源占用)"""
    system = SOPSystem(camera_id)
    
    import time
    fps_time = time.time()
    fps_cnt  = 0
    fps      = 0.0
    
    print("运行中... 按 Q 退出, 按 R 重置")
    while True:
        ret, frame = system.cap.read()
        if not ret:
            continue
        
        result = system.process_frame(frame)
        vis    = result['frame']
        
        fps_cnt += 1
        if time.time() - fps_time >= 1.0:
            fps     = fps_cnt / (time.time() - fps_time)
            fps_cnt = 0
            fps_time = time.time()
        
        cv2.putText(vis, f"FPS: {fps:.1f}", (vis.shape[1] - 130, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("SOP AI System", vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            system.engine.reset()
            print("[重置] SOP状态已清零")
    
    system.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SOP 行为分析系统")
    parser.add_argument('--camera',  type=int, default=0)
    parser.add_argument('--no-ui',   action='store_true', help="不使用PyQt5")
    args = parser.parse_args()
    
    if args.no_ui:
        run_headless(args.camera)
    else:
        run_with_ui(args.camera)