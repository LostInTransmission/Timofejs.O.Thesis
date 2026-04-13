import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6 import QtWidgets, QtCore
from widgets import CollapsibleBox, SyncGLViewWidget

class UILayout:
    def __init__(self, mw):
        self.mw = mw
        self.inputs = {}
        self.param_rows = {}
        self.vp_splits = []
        self.vp_multi = []
        self.algo_headers = {}
        self.setup_ui()

    def create_viewport(self, is_sync=False):
        view = SyncGLViewWidget() if is_sync else gl.GLViewWidget()
        view.setCameraPosition(distance=50)
        
        overlay = QtWidgets.QWidget(view)
        overlay.setStyleSheet("background: transparent;")
        lyt = QtWidgets.QHBoxLayout(overlay)
        lyt.setContentsMargins(5, 5, 5, 5)
        lyt.setSpacing(8)
        
        btn_top = QtWidgets.QPushButton("Top")
        btn_front = QtWidgets.QPushButton("Front")
        btn_right = QtWidgets.QPushButton("Right")
        btn_iso = QtWidgets.QPushButton("Iso")
        btn_home = QtWidgets.QPushButton("Home")
        for b in [btn_top, btn_front, btn_right, btn_iso, btn_home]:
            b.setFixedSize(50, 30)
            b.setStyleSheet("background-color: rgba(50,50,50,180); color: white; border-radius: 4px; font-size: 11px; font-weight: bold;")
            lyt.addWidget(b)
            
        overlay.adjustSize()
        overlay.move(10, 10)

        title_label = QtWidgets.QLabel(view)
        title_label.setStyleSheet("QLabel { color: #FFFF00; font-weight: bold; font-size: 16px; background-color: rgba(0,0,0,100); padding: 5px; border-radius: 3px; }")
        title_label.setVisible(is_sync)

        warning_label = QtWidgets.QLabel(view)
        warning_label.setStyleSheet("QLabel { color: #FF4444; font-weight: bold; font-size: 13px; background-color: rgba(50, 0, 0, 200); padding: 5px 10px; border-radius: 4px; border: 1px solid #FF0000; }")
        warning_label.setText("ATTENTION: Layer height is calculated based on the midpoint of all points (no tops found)")
        warning_label.setVisible(False)
        
        info_container = QtWidgets.QWidget(view)
        info_layout = QtWidgets.QVBoxLayout(info_container)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(2)

        btn_toggle_stats = QtWidgets.QPushButton("−")
        btn_toggle_stats.setFixedSize(20, 20)
        btn_toggle_stats.setStyleSheet("QPushButton { color: #FFF; background-color: rgba(0,0,0,150); border: 1px solid #444; border-radius: 3px; font-weight: bold; } QPushButton:hover { background-color: rgba(50,50,50,150); }")

        btn_hl = QtWidgets.QHBoxLayout()
        btn_hl.addStretch()
        btn_hl.addWidget(btn_toggle_stats)

        info_label = QtWidgets.QLabel()
        info_label.setStyleSheet("QLabel { color: #FFFFFF; background-color: rgba(0, 0, 0, 150); border: 1px solid #444; border-radius: 5px; padding: 8px; font-family: Consolas, monospace; font-size: 12px; }")
        info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)

        info_layout.addLayout(btn_hl)
        info_layout.addWidget(info_label)

        def toggle_stats():
            if info_label.isVisible():
                info_label.setVisible(False)
                btn_toggle_stats.setText("+")
            else:
                info_label.setVisible(True)
                btn_toggle_stats.setText("−")

        btn_toggle_stats.clicked.connect(toggle_stats)
        
        grid_layout = QtWidgets.QGridLayout(view)
        grid_layout.setContentsMargins(10, 10, 10, 10)
        
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(title_label, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        vbox.addWidget(warning_label, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        
        grid_layout.addLayout(vbox, 0, 0, alignment=QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
        grid_layout.addWidget(info_container, 0, 1, alignment=QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignRight)
        grid_layout.setColumnStretch(0, 1)
        
        grid = gl.GLGridItem(color=(255, 255, 255, 0))
        view.addItem(grid)
        scatter_valid = gl.GLScatterPlotItem(size=3)
        view.addItem(scatter_valid)
        scatter_noise_z = gl.GLScatterPlotItem(size=3, color=(1, 0, 0, 0.6))
        view.addItem(scatter_noise_z)
        scatter_noise_x = gl.GLScatterPlotItem(size=3, color=(1, 0, 1, 0.6))
        view.addItem(scatter_noise_x)
        scatter_apex = gl.GLScatterPlotItem(size=8, color=(1, 1, 0, 1), pxMode=True)
        view.addItem(scatter_apex)
        line_apex = gl.GLLinePlotItem(width=2, color=(1, 1, 0, 1))
        view.addItem(line_apex)
        scatter_l0 = gl.GLScatterPlotItem(size=2, color=(0.5, 0.5, 0.5, 0.3))
        view.addItem(scatter_l0)
        line_avg_h = gl.GLLinePlotItem(color=(0, 1, 1, 0.8), width=2, mode='lines')
        view.addItem(line_avg_h)
        scatter_spikes = gl.GLScatterPlotItem(size=16, color=(0.6, 0.0, 1.0, 1.0), pxMode=True)
        view.addItem(scatter_spikes)
        scatter_spikes.setVisible(False)
        
        return {
            'view': view, 'grid': grid, 'scatter_valid': scatter_valid,
            'scatter_noise_z': scatter_noise_z, 'scatter_noise_x': scatter_noise_x,
            'scatter_apex': scatter_apex, 'line_apex': line_apex, 'scatter_l0': scatter_l0,
            'line_avg_h': line_avg_h, 'scatter_spikes': scatter_spikes,
            'info_label': info_label, 'title_label': title_label, 
            'warning_label': warning_label, 'btn_top': btn_top, 'btn_front': btn_front, 'btn_right': btn_right, 'btn_iso': btn_iso, 'btn_home': btn_home
        }

    def setup_ui(self):
        central = QtWidgets.QWidget()
        self.mw.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        self.stacked_widget = QtWidgets.QStackedWidget()
        
        self.vp_main = self.create_viewport(is_sync=False)
        self.stacked_widget.addWidget(self.vp_main['view'])
        
        split_container = QtWidgets.QWidget()
        split_layout = QtWidgets.QVBoxLayout(split_container)
        split_layout.setContentsMargins(0, 0, 0, 0)
        split_layout.setSpacing(2)
        
        for _ in range(2):
            vp = self.create_viewport(is_sync=True)
            split_layout.addWidget(vp['view'])
            self.vp_splits.append(vp)
            
        for vp in self.vp_splits:
            vp['view'].cameraChanged.connect(lambda opts, v=vp['view']: self.mw.on_camera_changed(opts, v))
            
        self.stacked_widget.addWidget(split_container)

        self.vp_3d = self.create_viewport(is_sync=False)
        self.vp_3d['line_apex_all'] = gl.GLLinePlotItem(color=(1, 1, 0, 1), width=2, mode='lines')
        self.vp_3d['view'].addItem(self.vp_3d['line_apex_all'])
        self.stacked_widget.addWidget(self.vp_3d['view'])

        self.vp_intra_widget = pg.GraphicsLayoutWidget()
        self.stacked_widget.addWidget(self.vp_intra_widget)

        intra_layout_overlay = QtWidgets.QVBoxLayout(self.vp_intra_widget)
        intra_layout_overlay.setContentsMargins(10, 10, 10, 10)

        intra_info_container = QtWidgets.QWidget()
        intra_info_layout = QtWidgets.QVBoxLayout(intra_info_container)
        intra_info_layout.setContentsMargins(0, 0, 0, 0)
        intra_info_layout.setSpacing(2)

        self.btn_toggle_intra_stats = QtWidgets.QPushButton("−")
        self.btn_toggle_intra_stats.setFixedSize(20, 20)
        self.btn_toggle_intra_stats.setStyleSheet("QPushButton { color: #FFF; background-color: rgba(0,0,0,150); border: 1px solid #444; border-radius: 3px; font-weight: bold; } QPushButton:hover { background-color: rgba(50,50,50,150); }")
        intra_btn_hl = QtWidgets.QHBoxLayout()
        intra_btn_hl.addStretch()
        intra_btn_hl.addWidget(self.btn_toggle_intra_stats)
        self.intra_info_label = QtWidgets.QLabel()
        self.intra_info_label.setStyleSheet("QLabel { color: #FFFFFF; background-color: rgba(0, 0, 0, 150); border: 1px solid #444; border-radius: 5px; padding: 8px; font-family: Consolas, monospace; font-size: 12px; }")
        self.intra_info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop)
        intra_info_layout.addLayout(intra_btn_hl)
        intra_info_layout.addWidget(self.intra_info_label)
        intra_layout_overlay.addWidget(intra_info_container, alignment=QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignRight)

        def toggle_intra_stats():
            if self.intra_info_label.isVisible():
                self.intra_info_label.setVisible(False)
                self.btn_toggle_intra_stats.setText("+")
            else:
                self.intra_info_label.setVisible(True)
                self.btn_toggle_intra_stats.setText("−")

        self.btn_toggle_intra_stats.clicked.connect(toggle_intra_stats)

        self.plot_height = self.vp_intra_widget.addPlot(row=0, col=0, title="Absolute Height Profile")
        self.plot_height.setLabel('left', 'Abs Height Z', units='mm')
        self.plot_height.setLabel('bottom', 'Position Y', units='mm')
        self.plot_height.showGrid(x=True, y=True)
        self.plot_height.setYRange(-15, 15)
        
        self.plot_speed = self.vp_intra_widget.addPlot(row=1, col=0, title="Velocity Map")
        self.plot_speed.setLabel('left', 'Speed', units='%')
        self.plot_speed.setLabel('bottom', 'Position Y', units='mm')
        self.plot_speed.showGrid(x=True, y=True)
        self.plot_speed.setXLink(self.plot_height)
        self.plot_speed.setYRange(55, 145)

        self.curve_apexes_2d = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 120))
        self.plot_height.addItem(self.curve_apexes_2d)
        self.curve_target_2d = self.plot_height.plot(pen=pg.mkPen('g', width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.curve_segments_2d = self.plot_height.plot(pen=pg.mkPen('w', width=3))
        self.curve_speed_base = self.plot_speed.plot(pen=pg.mkPen('r', width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.curve_speed_step = self.plot_speed.plot(pen=pg.mkPen('c', width=3), fillLevel=100, brush=(0, 255, 255, 50))

        self.multi_container = QtWidgets.QWidget()
        multi_layout = QtWidgets.QVBoxLayout(self.multi_container)
        multi_layout.setContentsMargins(0, 0, 0, 0)
        multi_layout.setSpacing(2)
        
        for _ in range(4):
            vp = self.create_viewport(is_sync=True)
            multi_layout.addWidget(vp['view'])
            self.vp_multi.append(vp)
            
        for vp in self.vp_multi:
            vp['view'].cameraChanged.connect(lambda opts, v=vp['view']: self.mw.on_camera_changed(opts, v))
            
        self.stacked_widget.addWidget(self.multi_container)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(250)
        
        ctrl_panel = QtWidgets.QWidget()
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_panel)

        top_btn_layout = QtWidgets.QHBoxLayout()
        self.btn_inter = QtWidgets.QPushButton("Inter-Layer")
        self.btn_intra = QtWidgets.QPushButton("Intra-Layer")
        self.btn_3d = QtWidgets.QPushButton("3D")
        
        self.btn_inter.clicked.connect(lambda: self.mw.set_mode('INTER'))
        self.btn_intra.clicked.connect(lambda: self.mw.set_mode('INTRA'))
        self.btn_3d.clicked.connect(lambda: self.mw.set_mode('3D'))
        
        top_btn_layout.addWidget(self.btn_inter)
        top_btn_layout.addWidget(self.btn_intra)
        top_btn_layout.addWidget(self.btn_3d)
        
        ctrl_layout.addLayout(top_btn_layout)

        self.layer_ctrl_inter = QtWidgets.QWidget()
        layout_inter = QtWidgets.QHBoxLayout(self.layer_ctrl_inter)
        layout_inter.setContentsMargins(0, 0, 0, 0)
        self.cb_layer = QtWidgets.QSpinBox()
        if self.mw.layers:
            self.cb_layer.setRange(min(self.mw.layers), max(self.mw.layers))
            self.cb_layer.setValue(self.mw.layers[-1] if len(self.mw.layers)>0 else 0)
        self.cb_layer.setPrefix("Layer: ")
        self.cb_layer.valueChanged.connect(self.mw.on_layer_changed)
        layout_inter.addWidget(self.cb_layer)
        ctrl_layout.addWidget(self.layer_ctrl_inter)

        self.layer_ctrl_3d = QtWidgets.QWidget()
        layout_3d = QtWidgets.QHBoxLayout(self.layer_ctrl_3d)
        layout_3d.setContentsMargins(0, 0, 0, 0)
        self.cb_layer_min = QtWidgets.QSpinBox()
        self.cb_layer_max = QtWidgets.QSpinBox()
        if self.mw.layers:
            self.cb_layer_min.setRange(min(self.mw.layers), max(self.mw.layers))
            self.cb_layer_max.setRange(min(self.mw.layers), max(self.mw.layers))
            self.cb_layer_min.setValue(min(self.mw.layers))
            self.cb_layer_max.setValue(max(self.mw.layers))
        self.cb_layer_min.setPrefix("Min: ")
        self.cb_layer_max.setPrefix("Max: ")
        self.cb_layer_min.valueChanged.connect(lambda: self.mw._execute_refresh(self.mw.renderer.refresh_3d) if self.mw.current_view_mode == '3D' else None)
        self.cb_layer_max.valueChanged.connect(lambda: self.mw._execute_refresh(self.mw.renderer.refresh_3d) if self.mw.current_view_mode == '3D' else None)
        layout_3d.addWidget(self.cb_layer_min)
        layout_3d.addWidget(self.cb_layer_max)
        ctrl_layout.addWidget(self.layer_ctrl_3d)

        self.grp_intra = CollapsibleBox("Intra-Layer Settings")
        intra_layout = QtWidgets.QFormLayout()
        prof_intra_layout = QtWidgets.QHBoxLayout()
        self.cb_intra_profile = QtWidgets.QComboBox()
        self.btn_save_intra = QtWidgets.QPushButton("Save")
        self.btn_new_intra = QtWidgets.QPushButton("New")
        prof_intra_layout.addWidget(QtWidgets.QLabel("Profile:"))
        prof_intra_layout.addWidget(self.cb_intra_profile, stretch=1)
        prof_intra_layout.addWidget(self.btn_save_intra)
        prof_intra_layout.addWidget(self.btn_new_intra)
        intra_layout.addRow(prof_intra_layout)

        def add_intra_row(label_text, widget, tooltip):
            lbl = QtWidgets.QLabel(label_text)
            lbl.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            intra_layout.addRow(lbl, widget)

        self.spin_seg_len = QtWidgets.QDoubleSpinBox()
        self.spin_seg_len.setRange(1.0, 100.0)
        self.spin_seg_len.setValue(20.0)
        self.spin_seg_len.valueChanged.connect(self.mw.on_intra_param_changed)
        self.spin_kp = QtWidgets.QDoubleSpinBox()
        self.spin_kp.setRange(0.0, 100.0)
        self.spin_kp.setValue(10.0)
        self.spin_kp.valueChanged.connect(self.mw.on_intra_param_changed)
        self.spin_deadband = QtWidgets.QDoubleSpinBox()
        self.spin_deadband.setRange(0.0, 5.0)
        self.spin_deadband.setSingleStep(0.1)
        self.spin_deadband.setValue(0.2)
        self.spin_deadband.valueChanged.connect(self.mw.on_intra_param_changed)
        self.spin_clip_start = QtWidgets.QDoubleSpinBox()
        self.spin_clip_start.setRange(0.0, 2000.0)
        self.spin_clip_start.setValue(0.0)
        self.spin_clip_start.valueChanged.connect(self.mw.on_intra_param_changed)
        self.spin_clip_end = QtWidgets.QDoubleSpinBox()
        self.spin_clip_end.setRange(0.0, 2000.0)
        self.spin_clip_end.setValue(100.0)
        self.spin_clip_end.valueChanged.connect(self.mw.on_intra_param_changed)

        add_intra_row("Segment Len (mm):", self.spin_seg_len, "")
        add_intra_row("Kp:", self.spin_kp, "")
        add_intra_row("Deadband (±mm):", self.spin_deadband, "")
        add_intra_row("Clip Start Y:", self.spin_clip_start, "")
        add_intra_row("Clip End Y:", self.spin_clip_end, "")

        self.chk_intra_apexes = QtWidgets.QCheckBox("Raw Apexes (Yellow)")
        self.chk_intra_apexes.setChecked(True)
        self.chk_intra_apexes.stateChanged.connect(self.mw.update_intra_visibility)
        self.chk_intra_target = QtWidgets.QCheckBox("Calculated Target (Green)")
        self.chk_intra_target.setChecked(True)
        self.chk_intra_target.toggled.connect(lambda state: self.mw.on_target_mode_changed(state, 'calc'))
        
        given_target_layout = QtWidgets.QHBoxLayout()
        given_target_layout.setContentsMargins(0, 0, 0, 0)
        self.chk_intra_given = QtWidgets.QCheckBox("Given targ. (Green)")
        self.chk_intra_given.toggled.connect(lambda state: self.mw.on_target_mode_changed(state, 'given'))
        self.spin_given_h = QtWidgets.QDoubleSpinBox()
        self.spin_given_h.setRange(0.0, 10.0)
        self.spin_given_h.setSingleStep(0.1)
        self.spin_given_h.setPrefix("h: ")
        self.spin_given_h.setValue(1.0)
        self.spin_given_h.setEnabled(False)
        self.spin_given_h.valueChanged.connect(self.mw.on_intra_param_changed)
        given_target_layout.addWidget(self.chk_intra_given)
        given_target_layout.addWidget(self.spin_given_h)

        self.chk_intra_segments = QtWidgets.QCheckBox("Segment Avg (White)")
        self.chk_intra_segments.setChecked(True)
        self.chk_intra_segments.stateChanged.connect(self.mw.update_intra_visibility)
        self.chk_intra_speed = QtWidgets.QCheckBox("Speed Map (Cyan)")
        self.chk_intra_speed.setChecked(True)
        self.chk_intra_speed.stateChanged.connect(self.mw.update_intra_visibility)

        intra_layout.addRow(self.chk_intra_apexes)
        intra_layout.addRow(self.chk_intra_target)
        intra_layout.addRow(given_target_layout)
        intra_layout.addRow(self.chk_intra_segments)
        intra_layout.addRow(self.chk_intra_speed)
        self.grp_intra.content_layout.addLayout(intra_layout)
        self.grp_intra.setVisible(False)
        ctrl_layout.addWidget(self.grp_intra)

        self.grp_proc_params = CollapsibleBox("Process Parameters")
        form_proc = QtWidgets.QFormLayout()
        form_proc.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.grp_gen_params = CollapsibleBox("General Parameters")
        form_gen = QtWidgets.QFormLayout()
        form_gen.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        prof_filter_layout = QtWidgets.QHBoxLayout()
        self.cb_filter_profile = QtWidgets.QComboBox()
        self.btn_save_filter = QtWidgets.QPushButton("Save")
        self.btn_new_filter = QtWidgets.QPushButton("New")
        prof_filter_layout.addWidget(QtWidgets.QLabel("Profile:"))
        prof_filter_layout.addWidget(self.cb_filter_profile, stretch=1)
        prof_filter_layout.addWidget(self.btn_save_filter)
        prof_filter_layout.addWidget(self.btn_new_filter)
        form_gen.addRow(prof_filter_layout)

        self.grp_alg_params = CollapsibleBox("Filter Parameters")
        form_alg = QtWidgets.QFormLayout()
        form_alg.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        def create_param_row(key, meta, layout):
            default_val, min_v, max_v, step, tooltip, algo_tag = meta
            lbl = QtWidgets.QLabel(key)
            lbl.setToolTip(tooltip)
            if isinstance(default_val, int):
                s = QtWidgets.QSpinBox()
            else:
                s = QtWidgets.QDoubleSpinBox()
            s.setRange(min_v, max_v)
            s.setValue(self.mw.params[key])
            s.setSingleStep(step)
            s.setCorrectionMode(QtWidgets.QAbstractSpinBox.CorrectionMode.CorrectToNearestValue)
            s.valueChanged.connect(self.mw.update_params)
            
            btn_reset = QtWidgets.QPushButton("R")
            btn_reset.setFixedWidth(28)
            btn_reset.clicked.connect(lambda _, v=default_val, w=s: w.setValue(v))
            
            h_widget = QtWidgets.QWidget()
            h_layout = QtWidgets.QHBoxLayout(h_widget)
            h_layout.setContentsMargins(0, 0, 0, 0)
            h_layout.setSpacing(2)
            h_layout.addWidget(btn_reset)
            h_layout.addWidget(s)
            
            self.inputs[key] = s
            self.param_rows[key] = {'label': lbl, 'control': h_widget, 'tag': algo_tag}
            layout.addRow(lbl, h_widget)

        last_algo = None
        for key, meta in self.mw.param_meta.items():
            if 'PROCESS' in meta[5]:
                create_param_row(key, meta, form_proc)
            elif 'ALL' in meta[5]:
                create_param_row(key, meta, form_gen)
            else:
                algo = meta[5][0]
                if algo != last_algo:
                    if algo not in self.algo_headers:
                        self.algo_headers[algo] = []
                    if last_algo is not None:
                        line = QtWidgets.QFrame()
                        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
                        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
                        line.setStyleSheet("background-color: #555; margin-top: 5px; margin-bottom: 2px;")
                        form_alg.addRow(line)
                        self.algo_headers[algo].append(line)
                    lbl = QtWidgets.QLabel(f"--- {algo} ---")
                    lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    lbl.setStyleSheet("color: #AAA; font-weight: bold; padding-bottom: 2px;")
                    form_alg.addRow(lbl)
                    self.algo_headers[algo].append(lbl)
                    last_algo = algo
                create_param_row(key, meta, form_alg)

        self.grp_proc_params.content_layout.addLayout(form_proc)
        self.grp_gen_params.content_layout.addLayout(form_gen)
        self.grp_alg_params.content_layout.addLayout(form_alg)
        ctrl_layout.addWidget(self.grp_proc_params)
        ctrl_layout.addWidget(self.grp_gen_params)
        ctrl_layout.addWidget(self.grp_alg_params)

        grp_smooth = CollapsibleBox("Apex Smoothing Mode")
        smooth_layout = QtWidgets.QVBoxLayout()
        self.chk_split_mode = QtWidgets.QCheckBox("Enable Split Mode (Orig / SMA)")
        self.chk_split_mode.toggled.connect(self.mw.toggle_mode)
        smooth_layout.addWidget(self.chk_split_mode)
        param_layout = QtWidgets.QFormLayout()
        lbl_sma = QtWidgets.QLabel("SMA Window:")
        self.sma_win = QtWidgets.QSpinBox()
        self.sma_win.setRange(3, 51)
        self.sma_win.setSingleStep(2)
        self.sma_win.setValue(3)
        self.sma_win.valueChanged.connect(self.mw.on_layer_changed)
        param_layout.addRow(lbl_sma, self.sma_win)
        smooth_layout.addLayout(param_layout)
        grp_smooth.content_layout.addLayout(smooth_layout)
        ctrl_layout.addWidget(grp_smooth)

        grp_vis = CollapsibleBox("Visibility")
        vis_layout = QtWidgets.QVBoxLayout()
        self.chk_valid = QtWidgets.QCheckBox("Valid Points (Color)")
        self.chk_valid.setChecked(True)
        self.chk_valid.toggled.connect(self.mw.update_visibility)
        self.chk_apex = QtWidgets.QCheckBox("Apexes (Yellow)")
        self.chk_apex.setChecked(True)
        self.chk_apex.toggled.connect(self.mw.update_visibility)
        self.chk_line_apex = QtWidgets.QCheckBox("Apex Line (Yellow)")
        self.chk_line_apex.setChecked(True)
        self.chk_line_apex.toggled.connect(self.mw.update_visibility)
        self.chk_noise_z = QtWidgets.QCheckBox("Noise: Height (Red)")
        self.chk_noise_z.setChecked(True)
        self.chk_noise_z.toggled.connect(self.mw.update_visibility)
        self.chk_noise_x = QtWidgets.QCheckBox("Noise: Spatial (Magenta)")
        self.chk_noise_x.setChecked(True)
        self.chk_noise_x.toggled.connect(self.mw.update_visibility)
        self.chk_l0 = QtWidgets.QCheckBox("Substrate (Gray)")
        self.chk_l0.setChecked(True)
        self.chk_l0.toggled.connect(self.mw.update_visibility)
        self.chk_avg_h = QtWidgets.QCheckBox("Avg Height (Cyan Dashed)")
        self.chk_avg_h.setChecked(True)
        self.chk_avg_h.toggled.connect(self.mw.update_visibility)
        vis_layout.addWidget(self.chk_valid)
        vis_layout.addWidget(self.chk_apex)
        vis_layout.addWidget(self.chk_line_apex)
        vis_layout.addWidget(self.chk_noise_z)
        vis_layout.addWidget(self.chk_noise_x)
        vis_layout.addWidget(self.chk_l0)
        vis_layout.addWidget(self.chk_avg_h)
        grp_vis.content_layout.addLayout(vis_layout)
        ctrl_layout.addWidget(grp_vis)

        grp_view = CollapsibleBox("View Controls")
        v_layout = QtWidgets.QVBoxLayout()
        self.chk_ortho = QtWidgets.QCheckBox("Orthographic")
        self.chk_ortho.toggled.connect(self.mw.toggle_ortho)
        v_layout.addWidget(self.chk_ortho)
        self.chk_ortho.setChecked(True)
        grp_view.content_layout.addLayout(v_layout)
        ctrl_layout.addWidget(grp_view)

        ctrl_layout.addStretch()
        scroll_area.setWidget(ctrl_panel)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(scroll_area)
        splitter.addWidget(self.stacked_widget)
        splitter.setSizes([360, 1040])
        layout.addWidget(splitter, stretch=1)

        self.right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QHBoxLayout(self.right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        toggle_container = QtWidgets.QWidget()
        toggle_layout = QtWidgets.QVBoxLayout(toggle_container)
        toggle_layout.setContentsMargins(0, 0, 0, 10)
        toggle_layout.addStretch()
        
        self.btn_toggle_bench = QtWidgets.QToolButton()
        self.btn_toggle_bench.setText("📋")
        self.btn_toggle_bench.setFixedSize(28, 28)
        self.btn_toggle_bench.setStyleSheet("QToolButton { border: 1px solid #555; background-color: #333; border-radius: 3px; padding: 0px; font-size: 16px; margin-bottom: 5px; }")
        self.btn_toggle_bench.clicked.connect(self.mw.toggle_benchmark)
        toggle_layout.addWidget(self.btn_toggle_bench)

        self.btn_toggle_right = QtWidgets.QToolButton()
        self.btn_toggle_right.setArrowType(QtCore.Qt.ArrowType.LeftArrow)
        self.btn_toggle_right.setFixedSize(28, 28)
        self.btn_toggle_right.setStyleSheet("QToolButton { border: 1px solid #555; background-color: #333; border-radius: 3px; padding: 0px; }")
        self.btn_toggle_right.clicked.connect(self.mw.toggle_export_menu)
        toggle_layout.addWidget(self.btn_toggle_right)

        self.export_panel = QtWidgets.QFrame()
        self.export_panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.export_panel.setFixedWidth(150)
        self.export_panel.setVisible(False)
        self.export_panel.setStyleSheet("QFrame { background-color: #1e1e1e; border-left: 1px solid #444; }")
        export_layout = QtWidgets.QVBoxLayout(self.export_panel)
        
        lbl_menu = QtWidgets.QLabel("Main Menu")
        lbl_menu.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_menu.setStyleSheet("font-weight: bold; color: #FFF; border: none;")
        lbl_algo = QtWidgets.QLabel("Algorithm:")
        lbl_algo.setStyleSheet("color: #CCC; border: none; margin-top: 5px;")
        
        self.cb_algorithm = QtWidgets.QComboBox()
        self.cb_algorithm.addItems(["NoFilter", "DBSCAN", "Median", "SOR"])

        self.btn_multi_filter = QtWidgets.QPushButton("Multi-Filter Window")
        self.btn_multi_filter.setStyleSheet("background-color: #2b5b84; font-weight: bold; margin-top: 5px; padding: 5px;")

        self.btn_calc_speed = QtWidgets.QPushButton("Calculate Filter Speed")

        lbl_calib = QtWidgets.QLabel("Auto-Calibration")
        lbl_calib.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_calib.setStyleSheet("font-weight: bold; color: #FFF; border: none; margin-top: 15px;")

        self.spin_calib_iter = QtWidgets.QSpinBox()
        self.spin_calib_iter.setRange(1, 1000)
        self.spin_calib_iter.setValue(50)
        self.spin_calib_iter.setPrefix("Iters: ")

        self.chk_calib_all = QtWidgets.QCheckBox("All Layers")
        self.chk_calib_all.setChecked(True)

        hl_calib_layers = QtWidgets.QHBoxLayout()
        hl_calib_layers.setContentsMargins(0, 0, 0, 0)
        self.spin_calib_min = QtWidgets.QSpinBox()
        self.spin_calib_max = QtWidgets.QSpinBox()
        self.spin_calib_min.setEnabled(False)
        self.spin_calib_max.setEnabled(False)
        hl_calib_layers.addWidget(self.spin_calib_min)
        hl_calib_layers.addWidget(self.spin_calib_max)

        self.chk_calib_all.toggled.connect(lambda state: (self.spin_calib_min.setEnabled(not state), self.spin_calib_max.setEnabled(not state)))

        self.spin_calib_penalty = QtWidgets.QDoubleSpinBox()
        self.spin_calib_penalty.setRange(0.0, 10000.0)
        self.spin_calib_penalty.setValue(150.0)
        self.spin_calib_penalty.setSingleStep(10.0)
        self.spin_calib_penalty.setPrefix("Spike Penalty: ")

        self.btn_calib_start = QtWidgets.QPushButton("Start")
        self.btn_calib_stop = QtWidgets.QPushButton("Stop")
        self.btn_calib_stop.setEnabled(False)
        
        self.calib_progress = QtWidgets.QProgressBar()
        self.calib_progress.setRange(0, 100)
        self.calib_progress.setFixedHeight(10)
        self.calib_progress.setVisible(False)
        self.calib_progress.setStyleSheet("QProgressBar { border: none; background-color: #333; } QProgressBar::chunk { background-color: #FFA500; }")
        
        lbl_export_sec = QtWidgets.QLabel("Export")
        lbl_export_sec.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_export_sec.setStyleSheet("font-weight: bold; color: #FFF; border: none; margin-top: 15px;")
        
        self.btn_export_stats = QtWidgets.QPushButton("Export Filter Statistics")
        self.btn_export = QtWidgets.QPushButton("Export Data")
        
        export_layout.addWidget(lbl_menu)
        export_layout.addWidget(lbl_algo)
        export_layout.addWidget(self.cb_algorithm)
        export_layout.addWidget(self.btn_multi_filter)
        export_layout.addWidget(self.btn_calc_speed)
        export_layout.addWidget(lbl_calib)
        export_layout.addWidget(self.spin_calib_iter)
        export_layout.addWidget(self.chk_calib_all)
        export_layout.addLayout(hl_calib_layers)
        export_layout.addWidget(self.spin_calib_penalty)
        export_layout.addWidget(self.btn_calib_start)
        export_layout.addWidget(self.btn_calib_stop)
        export_layout.addWidget(self.calib_progress)
        export_layout.addStretch()
        export_layout.addWidget(lbl_export_sec)
        export_layout.addWidget(self.btn_export_stats)
        export_layout.addWidget(self.btn_export)

        right_layout.addWidget(toggle_container)
        right_layout.addWidget(self.export_panel)
        layout.addWidget(self.right_container)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(15)
        self.progress_bar.setStyleSheet("QProgressBar { border: none; background-color: #333; } QProgressBar::chunk { background-color: #4CAF50; }")
        self.mw.statusBar().addWidget(self.progress_bar, 1)

        self.dock_bench = QtWidgets.QDockWidget("Algorithm Benchmark", self.mw)
        self.dock_bench.setAllowedAreas(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea | QtCore.Qt.DockWidgetArea.TopDockWidgetArea)
        
        bench_widget = QtWidgets.QWidget()
        bench_layout = QtWidgets.QVBoxLayout(bench_widget)
        bench_layout.setContentsMargins(0, 0, 0, 0)
        
        self.table_bench = QtWidgets.QTableWidget()
        self.table_bench.setColumnCount(5)
        self.table_bench.setHorizontalHeaderLabels(["Algorithm", "Spikes", "FFT Noise (%)", "Lag-1 Autocorr", "Time (ms)"])
        self.table_bench.horizontalHeader().setStretchLastSection(True)
        self.table_bench.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_bench.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table_bench.verticalHeader().setVisible(False)
        
        self.table_bench_sma = QtWidgets.QTableWidget()
        self.table_bench_sma.setColumnCount(5)
        self.table_bench_sma.setHorizontalHeaderLabels(["Algorithm", "Spikes (SMA)", "FFT Noise (%)", "Lag-1 Autocorr", "Time (ms)"])
        self.table_bench_sma.horizontalHeader().setStretchLastSection(True)
        self.table_bench_sma.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_bench_sma.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table_bench_sma.verticalHeader().setVisible(False)
        self.table_bench_sma.setVisible(False)
        
        self.splitter_bench = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter_bench.addWidget(self.table_bench)
        self.splitter_bench.addWidget(self.table_bench_sma)
        
        bench_param_widget = QtWidgets.QWidget()
        bench_param_layout = QtWidgets.QHBoxLayout(bench_param_widget)
        bench_param_layout.setContentsMargins(5, 5, 5, 5)

        self.spin_spike_thr = QtWidgets.QDoubleSpinBox()
        self.spin_spike_thr.setRange(0.1, 10.0)
        self.spin_spike_thr.setSingleStep(0.1)
        self.spin_spike_thr.setValue(1.0)
        self.spin_spike_thr.setPrefix("Spike Thr: ")
        self.spin_spike_thr.valueChanged.connect(self.mw.on_bench_param_changed)
        
        self.spin_fft_cutoff = QtWidgets.QDoubleSpinBox()
        self.spin_fft_cutoff.setRange(0.01, 0.99)
        self.spin_fft_cutoff.setSingleStep(0.05)
        self.spin_fft_cutoff.setValue(0.25)
        self.spin_fft_cutoff.setPrefix("FFT Cutoff: ")
        self.spin_fft_cutoff.valueChanged.connect(self.mw.on_bench_param_changed)
        
        self.spin_lag_step = QtWidgets.QSpinBox()
        self.spin_lag_step.setRange(1, 50)
        self.spin_lag_step.setValue(1)
        self.spin_lag_step.setPrefix("Lag Step: ")
        self.spin_lag_step.valueChanged.connect(self.mw.on_bench_param_changed)

        self.chk_raw_apexes = QtWidgets.QCheckBox("Filtered Points Only")
        self.chk_raw_apexes.setChecked(False)
        self.chk_raw_apexes.toggled.connect(self.mw.on_raw_apexes_toggled)

        bench_param_layout.addWidget(self.spin_spike_thr)
        bench_param_layout.addWidget(self.spin_fft_cutoff)
        bench_param_layout.addWidget(self.spin_lag_step)
        bench_param_layout.addWidget(self.chk_raw_apexes)
        bench_param_layout.addStretch()
        
        bench_layout.addWidget(self.splitter_bench)
        bench_layout.addWidget(bench_param_widget)
        
        self.dock_bench.setWidget(bench_widget)
        self.mw.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_bench)
        self.dock_bench.hide()
        self.dock_bench.visibilityChanged.connect(self.mw.on_benchmark_visibility)