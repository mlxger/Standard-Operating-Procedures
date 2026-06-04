import sys
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class VideoThread(QThread):
    """摄像头采集 + AI推理线程"""
    frame_ready  = pyqtSignal(np.ndarray)
    state_update = pyqtSignal(dict)
    
    def __init__(self, system):
        super().__init__()
        self.system  = system
        self.running = True
        self._fps_counter = 0
        self._fps_start   = time.time()
        self.fps = 0.0
    
    def run(self):
        while self.running:
            ret, frame = self.system.cap.read()
            if not ret:
                continue
            
            results = self.system.process_frame(frame)
            self.frame_ready.emit(results['frame'])
            self.state_update.emit(results['sop_state'])
            
            self._fps_counter += 1
            if time.time() - self._fps_start >= 1.0:
                self.fps = self._fps_counter / (time.time() - self._fps_start)
                self._fps_counter = 0
                self._fps_start   = time.time()
    
    def stop(self):
        self.running = False
        self.wait()


class SOPDashboard(QMainWindow):
    def __init__(self, system):
        super().__init__()
        self.system = system
        self.setWindowTitle("AI行为监控系统")
        self.setMinimumSize(1400, 860)
        self._setup_ui()
        
        self.video_thread = VideoThread(system)
        self.video_thread.frame_ready.connect(self._update_video)
        self.video_thread.state_update.connect(self._update_state)
        self.video_thread.start()
    
    def _setup_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a2e; }
            QWidget     { background-color: #1a1a2e; color: #e0e0e0; }
            QLabel      { color: #e0e0e0; }
            QGroupBox   { border: 1px solid #444; border-radius: 6px;
                          margin-top: 8px; padding: 8px;
                          font-weight: bold; color: #00d4ff; }
            QTableWidget { background: #0d0d1f; gridline-color: #333;
                           border: none; }
            QTableWidget::item { padding: 4px; }
            QHeaderView::section { background: #252545; color: #00d4ff;
                                   border: 1px solid #444; padding: 4px; }
            QPushButton { background: #00a86b; color: white; border-radius: 4px;
                          padding: 8px 16px; font-weight: bold; }
            QPushButton:hover  { background: #00c07e; }
            QPushButton#stop   { background: #c0392b; }
            QPushButton#pause  { background: #e67e22; }
            QPushButton#reset  { background: #2980b9; }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # ── 顶栏 ──
        main_layout.addWidget(self._build_top_bar())
        
        # ── 中间内容区 ──
        content = QHBoxLayout()
        content.addWidget(self._build_video_panel(), 7)
        content.addWidget(self._build_right_panel(), 3)
        main_layout.addLayout(content)
        
        # ── 底部控制栏 ──
        main_layout.addWidget(self._build_bottom_bar())
    
    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet("background:#0d0d1f; border-radius:4px;")
        bar.setFixedHeight(50)
        layout = QHBoxLayout(bar)
        
        self.lbl_project  = QLabel("项目: 组装灯")
        self.lbl_operator = QLabel("作业员: 张三")
        self.lbl_mode     = QLabel("模式: 常规顺序")
        self.lbl_status   = QLabel("状态: 运行")
        self.lbl_elapsed  = QLabel("耗时: 00:00")
        
        for lbl in [self.lbl_project, self.lbl_operator,
                    self.lbl_mode,    self.lbl_status, self.lbl_elapsed]:
            lbl.setStyleSheet("color:#00d4ff; font-size:14px; padding:0 12px;")
            layout.addWidget(lbl)
        
        layout.addStretch()
        self.lbl_datetime = QLabel()
        self.lbl_datetime.setStyleSheet("color:#aaa; font-size:13px;")
        layout.addWidget(self.lbl_datetime)
        
        # 定时更新时间
        timer = QTimer(self)
        timer.timeout.connect(self._tick_clock)
        timer.start(1000)
        
        return bar
    
    def _tick_clock(self):
        from datetime import datetime
        self.lbl_datetime.setText(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _build_video_panel(self) -> QWidget:
        group = QGroupBox("实时监控")
        layout = QVBoxLayout(group)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#000; border-radius:4px;")
        layout.addWidget(self.video_label)
        
        return group
    
    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        
        # 统计卡片
        layout.addWidget(self._build_stats_card())
        # 步骤表格
        layout.addWidget(self._build_step_table())
        # 历史统计表
        layout.addWidget(self._build_history_table())
        
        return panel
    
    def _build_stats_card(self) -> QWidget:
        card = QWidget()
        card.setStyleSheet("background:#0d0d1f; border-radius:6px; padding:8px;")
        layout = QGridLayout(card)
        
        # 良率圆圈
        self.lbl_yield = QLabel("25.00%\n良率")
        self.lbl_yield.setAlignment(Qt.AlignCenter)
        self.lbl_yield.setStyleSheet(
            "color:#00d4ff; font-size:18px; font-weight:bold;"
            "border:3px solid #00d4ff; border-radius:50px;"
            "min-width:100px; min-height:100px;"
        )
        layout.addWidget(self.lbl_yield, 0, 0, 2, 1)
        
        def _stat_lbl(text, color):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"color:{color}; font-size:22px; font-weight:bold;")
            return lbl
        
        self.lbl_total   = _stat_lbl("0",  "#e0e0e0")
        self.lbl_good    = _stat_lbl("0",  "#00c07e")
        self.lbl_bad_act = _stat_lbl("0",  "#e74c3c")
        self.lbl_bad_pro = _stat_lbl("0",  "#e74c3c")
        
        layout.addWidget(self.lbl_total,   0, 1)
        layout.addWidget(self.lbl_good,    0, 2)
        layout.addWidget(self.lbl_bad_act, 0, 3)
        layout.addWidget(self.lbl_bad_pro, 0, 4)
        
        for i, txt in enumerate(["检测数量","良品数","动作不良","产品不良"]):
            lbl = QLabel(txt)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color:#888; font-size:11px;")
            layout.addWidget(lbl, 1, i+1)
        
        return card
    
    def _build_step_table(self) -> QWidget:
        group = QGroupBox("步骤状态")
        layout = QVBoxLayout(group)
        
        self.step_table = QTableWidget()
        self.step_table.setColumnCount(3)
        self.step_table.setHorizontalHeaderLabels(["步骤", "结果", "用时(s)"])
        self.step_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.step_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.step_table.setSelectionMode(QTableWidget.NoSelection)
        self.step_table.setFixedHeight(240)
        layout.addWidget(self.step_table)
        return group
    
    def _build_history_table(self) -> QWidget:
        group = QGroupBox("步骤统计")
        layout = QVBoxLayout(group)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(
            ["No", "步骤", "OK数", "NG数", "不良率"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.history_table)
        return group
    
    def _build_bottom_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(60)
        layout = QHBoxLayout(bar)
        
        btn_start = QPushButton("▶ 开始")
        btn_stop  = QPushButton("■ 停止"); btn_stop.setObjectName("stop")
        btn_pause = QPushButton("⏸ 待机"); btn_pause.setObjectName("pause")
        btn_reset = QPushButton("↺ 清零"); btn_reset.setObjectName("reset")
        
        btn_reset.clicked.connect(self._on_reset)
        
        for btn in [btn_start, btn_stop, btn_pause, btn_reset]:
            btn.setFixedSize(120, 44)
            layout.addWidget(btn)
        
        layout.addStretch()
        
        self.lbl_runtime = QLabel("持续运行: 00:00:00")
        self.lbl_runtime.setStyleSheet("color:#00d4ff; font-size:14px;")
        layout.addWidget(self.lbl_runtime)
        
        return bar
    
    # ── 槽函数 ──
    
    def _on_reset(self):
        self.system.sop_engine.reset()
    
    @pyqtSlot(np.ndarray)
    def _update_video(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg  = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix   = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(),
                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
    
    @pyqtSlot(dict)
    def _update_state(self, state: dict):
        # 统计卡片
        self.lbl_yield.setText(f"{state['yield_rate']:.1f}%\n良率")
        self.lbl_total.setText(str(state['total_detections']))
        self.lbl_good.setText(str(state['good_products']))
        self.lbl_bad_act.setText(str(state['bad_actions']))
        self.lbl_bad_pro.setText(str(state['bad_products']))
        
        # 步骤表
        steps = state['steps']
        self.step_table.setRowCount(len(steps))
        for i, step in enumerate(steps):
            self.step_table.setItem(i, 0, QTableWidgetItem(step.name))
            
            status_item = QTableWidgetItem(step.status.value)
            if step.status.value == 'PASS':
                status_item.setForeground(QColor('#00c07e'))
                status_item.setBackground(QColor('#0d2d1e'))
            elif step.status.value == 'FAIL':
                status_item.setForeground(QColor('#e74c3c'))
            elif step.status.value == 'Running':
                status_item.setForeground(QColor('#f39c12'))
            else:
                status_item.setForeground(QColor('#888'))
            self.step_table.setItem(i, 1, status_item)
            
            elapsed = f"{step.elapsed_time:.1f}" if step.elapsed_time > 0 else "0.0"
            self.step_table.setItem(i, 2, QTableWidgetItem(elapsed))
        
        # 历史统计表
        self.history_table.setRowCount(len(steps))
        for i, step in enumerate(steps):
            self.history_table.setItem(i, 0, QTableWidgetItem(str(step.step_id)))
            self.history_table.setItem(i, 1, QTableWidgetItem(step.name))
            self.history_table.setItem(i, 2, QTableWidgetItem(str(step.ok_count)))
            self.history_table.setItem(i, 3, QTableWidgetItem(str(step.ng_count)))
            ng_item = QTableWidgetItem(f"{step.defect_rate:.2f}%")
            if step.defect_rate > 0:
                ng_item.setForeground(QColor('#e74c3c'))
            self.history_table.setItem(i, 4, ng_item)
        
        # 运行时间
        e = int(state['elapsed_time'])
        self.lbl_runtime.setText(
            f"持续运行: {e//3600:02d}:{(e%3600)//60:02d}:{e%60:02d}"
        )
    
    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()