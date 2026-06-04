import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Optional
import yaml


class ObjectDetector:
    """YOLOv8-seg 目标检测 + 实例分割"""

    def __init__(self, config_path: str = "configs/model_config.yaml"):
        # ✅ 修复：添加 encoding='utf-8'
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)['yolo']

        self.model  = YOLO(cfg['model'])
        self.conf   = cfg['conf_threshold']
        self.iou    = cfg['iou_threshold']
        self.device = cfg['device']
        self.imgsz  = cfg['imgsz']

        self.colors = {
            'person': (0, 255, 0),
            'tool':   (0, 165, 255),
            'part':   (255, 0, 0),
        }
        print("[ObjectDetector] ✅ 初始化完成")
        print(f"  模型路径  : {self.model}")
        print(f"  iou  : {self.iou}")
        print(f"  检测置信度: {self.conf}")


    def detect(self, frame: np.ndarray) -> Dict:
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            masks = result.masks

            for i, box in enumerate(boxes):
                det = {
                    'bbox':       box.xyxy[0].cpu().numpy().astype(int),
                    'bbox_xywh':  box.xywh[0].cpu().numpy(),
                    'conf':       float(box.conf[0]),
                    'class_id':   int(box.cls[0]),
                    'class_name': self.model.names[int(box.cls[0])],
                    'mask':       masks.data[i].cpu().numpy() if masks else None
                }
                detections.append(det)

        return {'detections': detections, 'raw': results[0]}

    def draw(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        vis = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = self.colors.get(det['class_name'], (128, 128, 128))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = f"{det['class_name']} {det['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis, (x1, y1 - th - 8), (x1 + tw, y1), color, -1
            )
            cv2.putText(
                vis, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

            if det['mask'] is not None:
                mask_resized = cv2.resize(
                    det['mask'], (frame.shape[1], frame.shape[0])
                )
                overlay = vis.copy()
                overlay[mask_resized > 0.5] = color
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        return vis