import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import List, Dict
import yaml


class ObjectTracker:
    """DeepSORT 多目标追踪"""

    def __init__(self, config_path: str = "configs/model_config.yaml"):
        # ✅ 修复：添加 encoding='utf-8'
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)['tracker']

        self.tracker = DeepSort(
            max_age=cfg['max_age'],
            n_init=cfg['n_init'],
            nms_max_overlap=1.0,
            max_cosine_distance=cfg['max_cosine_distance'],
            nn_budget=100,
            embedder='mobilenet',
            half=True,
            bgr=True,
        )
        self.track_history: Dict[int, List] = {}
        print("[ObjectTracker] ✅ 初始化完成")

    def update(self, detections: List[Dict],
               frame: np.ndarray) -> List[Dict]:
        if not detections:
            return []

        raw_dets = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w, h = x2 - x1, y2 - y1
            raw_dets.append(
                ([x1, y1, w, h], det['conf'], det['class_name'])
            )

        tracks = self.tracker.update_tracks(raw_dets, frame=frame)

        tracked = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid  = track.track_id
            ltrb = track.to_ltrb().astype(int)

            cx = (ltrb[0] + ltrb[2]) // 2
            cy = (ltrb[1] + ltrb[3]) // 2
            if tid not in self.track_history:
                self.track_history[tid] = []
            self.track_history[tid].append((cx, cy))
            if len(self.track_history[tid]) > 50:
                self.track_history[tid].pop(0)

            tracked.append({
                'track_id':   tid,
                'bbox':       ltrb,
                'class_name': track.det_class,
                'history':    self.track_history[tid]
            })

        return tracked

    def draw_tracks(self, frame: np.ndarray,
                    tracks: List[Dict]) -> np.ndarray:
        vis = frame.copy()
        for t in tracks:
            x1, y1, x2, y2 = t['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                vis, f"ID:{t['track_id']}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )
            pts = t.get('history', [])
            for i in range(1, len(pts)):
                cv2.line(vis, pts[i-1], pts[i], (0, 255, 255), 1)
        return vis