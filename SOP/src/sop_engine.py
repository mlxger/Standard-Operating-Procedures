import time
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class StepStatus(Enum):
    IDLE    = "Idle"
    RUNNING = "Running"
    PASS    = "PASS"
    FAIL    = "FAIL"


@dataclass
class SOPStep:
    step_id:          int
    name:             str
    expected_actions: List[str]
    time_limit:       float
    status:           StepStatus = StepStatus.IDLE
    start_time:       Optional[float] = None
    elapsed_time:     float = 0.0
    ok_count:         int = 0
    ng_count:         int = 0
    pt_sum:           float = 0.0

    @property
    def defect_rate(self) -> float:
        total = self.ok_count + self.ng_count
        return round(self.ng_count / total * 100, 2) if total > 0 else 0.0

    @property
    def avg_pt(self) -> float:
        total = self.ok_count + self.ng_count
        return round(self.pt_sum / total, 1) if total > 0 else 0.0


class SOPEngine:
    def __init__(self, config_path: str = "configs/sop_steps.yaml"):
        self.steps: List[SOPStep] = []
        self.current_idx = 0
        self._load_config(config_path)
        self._reset_stats()

    def _load_config(self, path: str):
        # ✅ 修复：添加 encoding='utf-8'
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        self.project  = cfg.get('project', '未命名项目')
        self.operator = cfg.get('operator', '未知')
        self.mode     = cfg.get('mode', '常规顺序')

        for s in cfg['steps']:
            self.steps.append(SOPStep(
                step_id=s['id'],
                name=s['name'],
                expected_actions=s['expected_actions'],
                time_limit=s.get('time_limit', 30.0)
            ))

    def _reset_stats(self):
        self.total_detections = 0
        self.good_products    = 0
        self.bad_actions      = 0
        self.bad_products     = 0
        self.session_start    = time.time()

    def update(self, action_result: Optional[Dict]) -> Dict:
        if self.current_idx >= len(self.steps):
            return self._build_state()

        step        = self.steps[self.current_idx]
        action_name = action_result['action_name'] if action_result else None
        confidence  = action_result['confidence']  if action_result else 0.0

        if step.status == StepStatus.IDLE:
            if action_name in step.expected_actions and confidence >= 0.65:
                step.status     = StepStatus.RUNNING
                step.start_time = time.time()

        elif step.status == StepStatus.RUNNING:
            step.elapsed_time = time.time() - (step.start_time or time.time())

            if action_name in step.expected_actions and confidence >= 0.65:
                pt             = step.elapsed_time * 1000
                step.status    = StepStatus.PASS
                step.ok_count += 1
                step.pt_sum   += pt
                self.total_detections += 1
                self.good_products    += 1
                self._next_step()

            elif step.elapsed_time > step.time_limit:
                step.status    = StepStatus.FAIL
                step.ng_count += 1
                self.total_detections += 1
                self.bad_products += 1
                self.bad_actions  += 1

        return self._build_state()

    def _next_step(self):
        if self.current_idx + 1 < len(self.steps):
            self.current_idx += 1
            self.steps[self.current_idx].status     = StepStatus.IDLE
            self.steps[self.current_idx].start_time = None

    def _build_state(self) -> Dict:
        elapsed = time.time() - self.session_start
        total   = max(self.total_detections, 1)
        yield_r = round(self.good_products / total * 100, 1)

        return {
            'project':          self.project,
            'operator':         self.operator,
            'mode':             self.mode,
            'steps':            self.steps,
            'current_idx':      self.current_idx,
            'total_detections': self.total_detections,
            'good_products':    self.good_products,
            'bad_actions':      self.bad_actions,
            'bad_products':     self.bad_products,
            'yield_rate':       yield_r,
            'elapsed_time':     elapsed,
            'status':           '运行' if self.current_idx < len(self.steps) else '完成'
        }

    def reset(self):
        for step in self.steps:
            step.status       = StepStatus.IDLE
            step.start_time   = None
            step.elapsed_time = 0.0
        self.current_idx = 0
        self._reset_stats()