import sys
import os
import json
import csv
import time
from datetime import datetime
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore
from config import *
from processing import Processor, MetricsEvaluator
from ui_layout import UILayout
from render import RenderLogic

class CalibrationThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int)
    finished_calib = QtCore.pyqtSignal(dict, float)

    def __init__(self, df, layers, algo, base_params, meta, iters, use_raw, spike_thr, fft, lag, spike_penalty):
        super().__init__()
        self.df = df
        self.layers = layers
        self.algo = algo
        self.base_params = base_params
        self.meta = meta
        self.iters = iters
        self.use_raw = use_raw
        self.spike_thr = spike_thr
        self.fft = fft
        self.lag = lag
        self.spike_penalty = spike_penalty
        self.running = True

    def run(self):
        best_score = -float('inf')
        best_params = {}

        for i in range(self.iters):
            if not self.running: break

            test_params = self.base_params.copy()
            for k, meta in self.meta.items():
                tags = meta[5]
                if self.algo in tags and 'ALL' not in tags and 'PROCESS' not in tags:
                    min_v, max_v = meta[1], meta[2]
                    if isinstance(meta[0], int):
                        test_params[k] = np.random.randint(int(min_v), int(max_v) + 1)
                    else:
                        test_params[k] = round(np.random.uniform(min_v, max_v), 2)

            tot_val, tot_sp = 0, 0
            tot_fft = 0.0
            valid_l = 0

            for lid in self.layers:
                if not self.running: break
                target, _ = Processor.prepare_layer(self.df, lid)
                if target is None or target.empty: continue

                valid, _, _, _ = Processor.filter_data(target, test_params, self.algo)
                if valid is None or valid.empty: continue

                if self.use_raw:
                    apexes = Processor.calc_raw_apexes(valid)
                    tot_val += len(apexes)
                else:
                    apexes, _, _, stats = Processor.calc_apexes(valid, test_params)
                    tot_val += stats['valid']

                sp, f_rat, _, _ = MetricsEvaluator.evaluate(apexes, self.spike_thr, self.fft, self.lag)
                tot_sp += sp
                tot_fft += f_rat
                valid_l += 1

            if not self.running: break

            if valid_l > 0:
                avg_val = tot_val / valid_l
                avg_sp = tot_sp / valid_l
                avg_fft = tot_fft / valid_l
                
                score = avg_val - (avg_sp * self.spike_penalty) - (avg_fft * 10.0)
            else:
                score = -float('inf')

            if score > best_score:
                best_score = score
                best_params = test_params.copy()

            self.progress.emit(i + 1, self.iters)

        if self.running:
            self.finished_calib.emit(best_params, best_score)

    def stop(self):
        self.running = False

class VisualizerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Layer Analyzer")
        self.resize(1400, 900)
        
        self.layer_cache = {}
        self.last_params = None
        self.last_layer = None
        self.current_view_mode = 'INTER'
        self.active_3d_lines = 0
        self.current_algorithm = 'NoFilter'
        self._is_working = False
        self._refresh_queued = False
        self.calib_thread = None
        self.filter_times = {a: "-" for a in ["NoFilter", "DBSCAN", "Median", "SOR"]}

        self.full_df = Processor.load_data(JOB_LOG_FILE)
        if self.full_df is not None:
            self.layers = sorted(self.full_df['Layer'].unique())
        else:
            self.layers = [0]

        self.params = {
            'h': 0.9,
            'TRIM_START': DEF_TRIM_START, 'TRIM_END': DEF_TRIM_END,
            'X_MIN': DEF_X_MIN, 'X_MAX': DEF_X_MAX,
            'PERC_MIN': DEF_PERC_MIN, 'DELTA_H': DEF_DELTA_H,
            'X_TOL': DEF_X_TOL,
            'MIN_PTS': DEF_MIN_PTS, 'MIN_WIDTH': DEF_MIN_WIDTH,
            'MIN_CURV': DEF_MIN_CURV, 'APEX_TOL': DEF_APEX_TOL,
            'CONN_DX': DEF_CONN_DX, 'CONN_DY': DEF_CONN_DY, 'CONN_DZ': DEF_CONN_DZ,
            'MIN_SAMPLES': DEF_MIN_SAMPLES, 'MIN_CLUSTER_SIZE': DEF_MIN_CLUSTER_SIZE, 'EPS': DEF_EPS,
            'MEDIAN_WINDOW': DEF_MEDIAN_WINDOW, 'MEDIAN_THRESHOLD': DEF_MEDIAN_THRESHOLD,
            'SOR_K': DEF_SOR_K, 'SOR_MULT': DEF_SOR_MULT
        }
        
        self.param_meta = {
            'h': (0.9, -10.0, 10.0, 0.1, "Sensor lift step per layer", ['PROCESS']),
            'TRIM_START': (DEF_TRIM_START, 0, 500, 1.0, "", ['ALL']),
            'TRIM_END': (DEF_TRIM_END, 0, 500, 1.0, "", ['ALL']),
            'X_MIN': (DEF_X_MIN, -50, 50, 0.1, "", ['ALL']),
            'X_MAX': (DEF_X_MAX, -50, 50, 0.1, "", ['ALL']),
            'PERC_MIN': (DEF_PERC_MIN, 0, 100, 1, "", ['ALL']),
            'DELTA_H': (DEF_DELTA_H, 0, 10, 0.1, "", ['ALL']),
            'X_TOL': (DEF_X_TOL, 0.1, 10, 0.1, "", ['ALL']),
            'MIN_PTS': (DEF_MIN_PTS, 3, 100, 1, "", ['ALL']),
            'MIN_WIDTH': (DEF_MIN_WIDTH, 0, 10, 0.1, "", ['ALL']),
            'MIN_CURV': (DEF_MIN_CURV, -10, 10, 0.01, "", ['ALL']),
            'APEX_TOL': (DEF_APEX_TOL, 0.1, 20, 0.1, "", ['ALL']),
            'CONN_DX': (DEF_CONN_DX, 0.01, 10.0, 0.01, "", ['DBSCAN']),
            'CONN_DY': (DEF_CONN_DY, 0.01, 50.0, 0.1, "", ['DBSCAN']),
            'CONN_DZ': (DEF_CONN_DZ, 0.01, 10.0, 0.01, "", ['DBSCAN']),
            'MIN_SAMPLES': (DEF_MIN_SAMPLES, 1, 50, 1, "", ['DBSCAN']),
            'MIN_CLUSTER_SIZE': (DEF_MIN_CLUSTER_SIZE, 1, 100, 1, "", ['DBSCAN']),
            'EPS': (DEF_EPS, 0.1, 10.0, 0.1, "", ['DBSCAN']),
            'MEDIAN_WINDOW': (DEF_MEDIAN_WINDOW, 3, 51, 2, "", ['Median']),
            'MEDIAN_THRESHOLD': (DEF_MEDIAN_THRESHOLD, 0.01, 5.0, 0.01, "", ['Median']),
            'SOR_K': (DEF_SOR_K, 2, 100, 1, "", ['SOR']),
            'SOR_MULT': (DEF_SOR_MULT, 0.1, 5.0, 0.1, "", ['SOR'])
        }

        self.ui = UILayout(self)
        self.renderer = RenderLogic(self)

        for vp in [self.ui.vp_main, self.ui.vp_3d] + self.ui.vp_splits + self.ui.vp_multi:
            vp['btn_top'].clicked.connect(lambda _, v=vp['view']: self.set_cam(0, 90, v))
            vp['btn_front'].clicked.connect(lambda _, v=vp['view']: self.set_cam(0, 0, v))
            vp['btn_right'].clicked.connect(lambda _, v=vp['view']: self.set_cam(-90, 0, v))
            vp['btn_iso'].clicked.connect(lambda _, v=vp['view']: self.set_cam(45, 30, v))
            if 'btn_home' in vp:
                vp['btn_home'].clicked.connect(lambda _, v=vp['view']: self.reset_cam(v))

        self.PROFILE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles.json")
        self.profiles = {'filter': {}, 'intra': {}}
        self.load_profiles()
        self.refresh_profile_combos()

        if self.layers:
            self.ui.spin_calib_min.setRange(min(self.layers), max(self.layers))
            self.ui.spin_calib_max.setRange(min(self.layers), max(self.layers))
            self.ui.spin_calib_min.setValue(min(self.layers))
            self.ui.spin_calib_max.setValue(max(self.layers))

        self.ui.cb_filter_profile.currentTextChanged.connect(self.apply_filter_profile)
        self.ui.btn_save_filter.clicked.connect(lambda: self.save_filter_profile(is_new=False))
        self.ui.btn_new_filter.clicked.connect(lambda: self.save_filter_profile(is_new=True))
        self.ui.cb_intra_profile.currentTextChanged.connect(self.apply_intra_profile)
        self.ui.btn_save_intra.clicked.connect(lambda: self.save_intra_profile(is_new=False))
        self.ui.btn_new_intra.clicked.connect(lambda: self.save_intra_profile(is_new=True))
        self.ui.cb_algorithm.currentTextChanged.connect(self.on_algorithm_changed)
        self.ui.btn_multi_filter.clicked.connect(lambda: self.set_mode('MULTI'))
        self.ui.btn_export_stats.clicked.connect(self.export_filter_statistics)
        self.ui.btn_export.clicked.connect(self.export_data)
        self.ui.btn_calib_start.clicked.connect(self.start_calibration)
        self.ui.btn_calib_stop.clicked.connect(self.stop_calibration)
        self.ui.btn_calc_speed.clicked.connect(self.calculate_filter_speed)

        self._is_initialized = True
        self.update_dynamic_params()
        if self.full_df is not None:
            self.set_mode('INTER')

    def show_loading(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.ui.progress_bar.setVisible(True)
        self.ui.progress_bar.setValue(0)
        if hasattr(self, 'renderer'):
            self.renderer.last_pct = -1
            
        sz = self.size()
        self.resize(sz.width(), sz.height() + 1)
        self.resize(sz)
        
        QtWidgets.QApplication.processEvents()

    def hide_loading(self):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.ui.progress_bar.setVisible(False)
        
        sz = self.size()
        self.resize(sz.width(), sz.height() - 1)
        self.resize(sz)

    def _execute_refresh(self, refresh_func):
        if getattr(self, '_is_working', False):
            self._refresh_queued = True
            return
        self._is_working = True
        while True:
            self._refresh_queued = False
            self.show_loading()
            try:
                refresh_func()
                if self.ui.dock_bench.isVisible():
                    self.run_benchmark()
            except:
                pass
            finally:
                self.hide_loading()
            if not getattr(self, '_refresh_queued', False):
                break
        self._is_working = False

    def update_dynamic_params(self):
        algo = self.current_algorithm
        is_multi = getattr(self, 'current_view_mode', 'INTER') == 'MULTI'
        for key, row in self.ui.param_rows.items():
            if 'ALL' in row['tag'] or 'PROCESS' in row['tag']:
                is_visible = True
            else:
                is_visible = is_multi or (algo in row['tag'])
                
            row['label'].setVisible(is_visible)
            row['control'].setVisible(is_visible)
                
        if hasattr(self.ui, 'algo_headers'):
            for a_name, widgets in self.ui.algo_headers.items():
                is_visible = is_multi or (algo == a_name)
                for w in widgets:
                    w.setVisible(is_visible)

    def load_profiles(self):
        if os.path.exists(self.PROFILE_FILE):
            try:
                with open(self.PROFILE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'filter' in data: self.profiles['filter'] = data['filter']
                    if 'intra' in data: self.profiles['intra'] = data['intra']
            except:
                pass
        if "Default" not in self.profiles['filter']:
            self.profiles['filter']["Default"] = {k: v[0] for k, v in self.param_meta.items()}
            self.profiles['filter']["Default"]['sma_win'] = 3
            self.profiles['filter']["Default"]['spike_thr'] = DEF_EVAL_SPIKE_THR
            self.profiles['filter']["Default"]['fft_cutoff'] = DEF_EVAL_FFT_CUTOFF
            self.profiles['filter']["Default"]['lag_step'] = DEF_EVAL_LAG
        if "Default" not in self.profiles['intra']:
            self.profiles['intra']["Default"] = {
                'seg_len': 20.0, 'kp': 10.0, 'deadband': 0.2, 
                'clip_start': 0.0, 'clip_end': 100.0, 'given_h': 1.0,
                'target_calc': True, 'target_given': False
            }

    def save_profiles_to_disk(self):
        try:
            with open(self.PROFILE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.profiles, f, indent=4)
        except:
            pass

    def refresh_profile_combos(self):
        self.ui.cb_filter_profile.blockSignals(True)
        curr_f = self.ui.cb_filter_profile.currentText()
        self.ui.cb_filter_profile.clear()
        self.ui.cb_filter_profile.addItems(list(self.profiles['filter'].keys()))
        if curr_f in self.profiles['filter']:
            self.ui.cb_filter_profile.setCurrentText(curr_f)
        elif "Default" in self.profiles['filter']:
            self.ui.cb_filter_profile.setCurrentText("Default")
        self.ui.cb_filter_profile.blockSignals(False)

        self.ui.cb_intra_profile.blockSignals(True)
        curr_i = self.ui.cb_intra_profile.currentText()
        self.ui.cb_intra_profile.clear()
        self.ui.cb_intra_profile.addItems(list(self.profiles['intra'].keys()))
        if curr_i in self.profiles['intra']:
            self.ui.cb_intra_profile.setCurrentText(curr_i)
        elif "Default" in self.profiles['intra']:
            self.ui.cb_intra_profile.setCurrentText("Default")
        self.ui.cb_intra_profile.blockSignals(False)

    def apply_filter_profile(self, name):
        if name not in self.profiles['filter']: return
        prof = self.profiles['filter'][name]
        for k in self.params:
            if k in prof and k in self.ui.inputs:
                self.ui.inputs[k].blockSignals(True)
                self.ui.inputs[k].setValue(prof[k])
                self.params[k] = prof[k]
                self.ui.inputs[k].blockSignals(False)
        if 'sma_win' in prof:
            self.ui.sma_win.blockSignals(True)
            self.ui.sma_win.setValue(prof['sma_win'])
            self.ui.sma_win.blockSignals(False)
        if 'spike_thr' in prof:
            self.ui.spin_spike_thr.blockSignals(True)
            self.ui.spin_spike_thr.setValue(prof['spike_thr'])
            self.ui.spin_spike_thr.blockSignals(False)
        if 'fft_cutoff' in prof:
            self.ui.spin_fft_cutoff.blockSignals(True)
            self.ui.spin_fft_cutoff.setValue(prof['fft_cutoff'])
            self.ui.spin_fft_cutoff.blockSignals(False)
        if 'lag_step' in prof:
            self.ui.spin_lag_step.blockSignals(True)
            self.ui.spin_lag_step.setValue(prof['lag_step'])
            self.ui.spin_lag_step.blockSignals(False)
            
        self.update_params()

    def apply_intra_profile(self, name):
        if name not in self.profiles['intra']: return
        prof = self.profiles['intra'][name]
        def safe_set(widget, val):
            widget.blockSignals(True)
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(val)
            else:
                widget.setValue(val)
            widget.blockSignals(False)
        if 'seg_len' in prof: safe_set(self.ui.spin_seg_len, prof['seg_len'])
        if 'kp' in prof: safe_set(self.ui.spin_kp, prof['kp'])
        if 'deadband' in prof: safe_set(self.ui.spin_deadband, prof['deadband'])
        if 'clip_start' in prof: safe_set(self.ui.spin_clip_start, prof['clip_start'])
        if 'clip_end' in prof: safe_set(self.ui.spin_clip_end, prof['clip_end'])
        if 'given_h' in prof: safe_set(self.ui.spin_given_h, prof['given_h'])
        if 'target_calc' in prof: safe_set(self.ui.chk_intra_target, prof['target_calc'])
        if 'target_given' in prof: safe_set(self.ui.chk_intra_given, prof['target_given'])
        self.ui.spin_given_h.setEnabled(self.ui.chk_intra_given.isChecked())
        self.update_intra_visibility()
        self.on_intra_param_changed()

    def save_filter_profile(self, is_new=False):
        if is_new:
            name, ok = QtWidgets.QInputDialog.getText(self, "New Filter Profile", "Enter profile name:")
            if not ok or not name.strip(): return
            name = name.strip()
        else:
            name = self.ui.cb_filter_profile.currentText()
            if not name: return
        prof = {k: w.value() for k, w in self.ui.inputs.items()}
        prof['sma_win'] = self.ui.sma_win.value()
        prof['spike_thr'] = self.ui.spin_spike_thr.value()
        prof['fft_cutoff'] = self.ui.spin_fft_cutoff.value()
        prof['lag_step'] = self.ui.spin_lag_step.value()
        self.profiles['filter'][name] = prof
        self.save_profiles_to_disk()
        self.refresh_profile_combos()
        self.ui.cb_filter_profile.setCurrentText(name)

    def save_intra_profile(self, is_new=False):
        if is_new:
            name, ok = QtWidgets.QInputDialog.getText(self, "New Intra Profile", "Enter profile name:")
            if not ok or not name.strip(): return
            name = name.strip()
        else:
            name = self.ui.cb_intra_profile.currentText()
            if not name: return
        prof = {
            'seg_len': self.ui.spin_seg_len.value(),
            'kp': self.ui.spin_kp.value(),
            'deadband': self.ui.spin_deadband.value(),
            'clip_start': self.ui.spin_clip_start.value(),
            'clip_end': self.ui.spin_clip_end.value(),
            'given_h': self.ui.spin_given_h.value(),
            'target_calc': self.ui.chk_intra_target.isChecked(),
            'target_given': self.ui.chk_intra_given.isChecked()
        }
        self.profiles['intra'][name] = prof
        self.save_profiles_to_disk()
        self.refresh_profile_combos()
        self.ui.cb_intra_profile.setCurrentText(name)

    def on_algorithm_changed(self, algo_name):
        self.current_algorithm = algo_name
        self.layer_cache.clear()
        self.update_dynamic_params()
        self.update_params()

    def set_cam(self, az, el, view):
        view.setCameraPosition(elevation=el, azimuth=az)
        if self.current_view_mode == 'INTER' and self.ui.chk_split_mode.isChecked():
            is_split_view = any(vp['view'] == view for vp in self.ui.vp_splits)
            if is_split_view:
                for vp in self.ui.vp_splits:
                    vp['view'].setCameraPosition(elevation=el, azimuth=az)
        elif self.current_view_mode == 'MULTI':
            is_multi_view = any(vp['view'] == view for vp in self.ui.vp_multi)
            if is_multi_view:
                for vp in self.ui.vp_multi:
                    vp['view'].setCameraPosition(elevation=el, azimuth=az)

    def reset_cam(self, view):
        view.setCameraPosition(distance=50, elevation=30, azimuth=45)
        if self.current_view_mode == 'INTER' and self.ui.chk_split_mode.isChecked():
            is_split_view = any(vp['view'] == view for vp in self.ui.vp_splits)
            if is_split_view:
                for vp in self.ui.vp_splits:
                    vp['view'].setCameraPosition(distance=50, elevation=30, azimuth=45)
        elif self.current_view_mode == 'MULTI':
            is_multi_view = any(vp['view'] == view for vp in self.ui.vp_multi)
            if is_multi_view:
                for vp in self.ui.vp_multi:
                    vp['view'].setCameraPosition(distance=50, elevation=30, azimuth=45)

    def on_camera_changed(self, opts, source_view):
        views_to_sync = []
        if self.current_view_mode == 'INTER' and self.ui.chk_split_mode.isChecked():
            views_to_sync = [vp['view'] for vp in self.ui.vp_splits]
        elif self.current_view_mode == 'MULTI':
            views_to_sync = [vp['view'] for vp in self.ui.vp_multi]
            
        for v in views_to_sync:
            if v is not source_view:
                v.sync_from(opts)

    def set_mode(self, mode):
        self.current_view_mode = mode
        active_style = "background-color: #4CAF50; color: white; font-weight: bold;"
        self.ui.btn_inter.setStyleSheet(active_style if mode == 'INTER' else "")
        self.ui.btn_intra.setStyleSheet(active_style if mode == 'INTRA' else "")
        self.ui.btn_3d.setStyleSheet(active_style if mode == '3D' else "")
        if hasattr(self.ui, 'btn_multi_filter'):
            self.ui.btn_multi_filter.setStyleSheet(active_style if mode == 'MULTI' else "background-color: #2b5b84; font-weight: bold; margin-top: 5px; padding: 5px;")
            
        self.ui.layer_ctrl_inter.setVisible(mode in ['INTER', 'INTRA', 'MULTI'])
        self.ui.layer_ctrl_3d.setVisible(mode == '3D')
        self.ui.grp_intra.setVisible(mode == 'INTRA')
        
        self.update_dynamic_params()
        
        if mode == 'INTER':
            self.ui.stacked_widget.setCurrentIndex(1 if self.ui.chk_split_mode.isChecked() else 0)
            self._execute_refresh(self.renderer.refresh)
        elif mode == 'INTRA':
            self.ui.stacked_widget.setCurrentIndex(3)
            self._execute_refresh(self.renderer.refresh_intra)
        elif mode == '3D':
            self.ui.stacked_widget.setCurrentIndex(2)
            self._execute_refresh(self.renderer.refresh_3d)
        elif mode == 'MULTI':
            self.ui.stacked_widget.setCurrentIndex(4)
            for vp in self.ui.vp_multi:
                vp['view'].setCameraPosition(elevation=0, azimuth=0)
            self._execute_refresh(self.renderer.refresh_multi)

    def toggle_mode(self, state):
        if self.current_view_mode == 'INTER':
            if state:
                self.ui.stacked_widget.setCurrentIndex(1)
                for vp in self.ui.vp_splits:
                    vp['view'].setCameraPosition(elevation=0, azimuth=0)
            else:
                self.ui.stacked_widget.setCurrentIndex(0)
            if self.ui.dock_bench.isVisible():
                self.ui.table_bench_sma.setVisible(state)
            self._execute_refresh(self.renderer.refresh)
        elif self.current_view_mode == 'INTRA':
            self._execute_refresh(self.renderer.refresh_intra)
        elif self.current_view_mode == '3D':
            self._execute_refresh(self.renderer.refresh_3d)

    def on_layer_changed(self):
        if self.current_view_mode == 'INTER':
            self._execute_refresh(self.renderer.refresh)
        elif self.current_view_mode == 'INTRA':
            self._execute_refresh(self.renderer.refresh_intra)
        elif self.current_view_mode == 'MULTI':
            self._execute_refresh(self.renderer.refresh_multi)

    def on_target_mode_changed(self, state, source):
        if state:
            if source == 'calc':
                self.ui.chk_intra_given.blockSignals(True)
                self.ui.chk_intra_given.setChecked(False)
                self.ui.chk_intra_given.blockSignals(False)
                self.ui.spin_given_h.setEnabled(False)
            else:
                self.ui.chk_intra_target.blockSignals(True)
                self.ui.chk_intra_target.setChecked(False)
                self.ui.chk_intra_target.blockSignals(False)
                self.ui.spin_given_h.setEnabled(True)
        self.update_intra_visibility()
        self._execute_refresh(self.renderer.refresh_intra)

    def on_intra_param_changed(self):
        if self.current_view_mode == 'INTRA':
            self._execute_refresh(self.renderer.refresh_intra)

    def update_params(self):
        if getattr(self, '_is_initialized', False) is False:
            return
        for k, w in self.ui.inputs.items():
            self.params[k] = w.value()
        if self.current_view_mode == 'INTER':
            self._execute_refresh(self.renderer.refresh)
        elif self.current_view_mode == '3D':
            self._execute_refresh(self.renderer.refresh_3d)
        elif self.current_view_mode == 'INTRA':
            self._execute_refresh(self.renderer.refresh_intra)
        elif self.current_view_mode == 'MULTI':
            self._execute_refresh(self.renderer.refresh_multi)

    def on_bench_param_changed(self):
        if getattr(self, '_is_initialized', False) is False:
            return
        if self.current_view_mode == 'INTER':
            self._execute_refresh(self.renderer.refresh)
        elif self.current_view_mode == '3D':
            self._execute_refresh(self.renderer.refresh_3d)
        elif self.current_view_mode == 'MULTI':
            self._execute_refresh(self.renderer.refresh_multi)

    def on_raw_apexes_toggled(self, state):
        if getattr(self, '_is_initialized', False) is False:
            return
            
        ignored_params = ['MIN_PTS', 'MIN_WIDTH', 'MIN_CURV', 'APEX_TOL']
        for p in ignored_params:
            if p in self.ui.inputs:
                self.ui.inputs[p].setEnabled(not state)

        self.layer_cache.clear()
        if self.current_view_mode == 'INTER':
            self._execute_refresh(self.renderer.refresh)
        elif self.current_view_mode == '3D':
            self._execute_refresh(self.renderer.refresh_3d)
        elif self.current_view_mode == 'INTRA':
            self._execute_refresh(self.renderer.refresh_intra)
        elif self.current_view_mode == 'MULTI':
            self._execute_refresh(self.renderer.refresh_multi)

    def update_intra_visibility(self):
        self.ui.curve_apexes_2d.setVisible(self.ui.chk_intra_apexes.isChecked())
        show_target = self.ui.chk_intra_target.isChecked() or self.ui.chk_intra_given.isChecked()
        self.ui.curve_target_2d.setVisible(show_target)
        self.ui.curve_segments_2d.setVisible(self.ui.chk_intra_segments.isChecked())
        self.ui.curve_speed_step.setVisible(self.ui.chk_intra_speed.isChecked())

    def update_visibility(self):
        if not hasattr(self, 'ui'): return
        vps = [self.ui.vp_main, self.ui.vp_3d] + self.ui.vp_splits + getattr(self.ui, 'vp_multi', [])
        for vp in vps:
            vp['scatter_valid'].setVisible(self.ui.chk_valid.isChecked())
            vp['scatter_apex'].setVisible(self.ui.chk_apex.isChecked())
            vp['line_apex'].setVisible(self.ui.chk_line_apex.isChecked())
            vp['scatter_noise_z'].setVisible(self.ui.chk_noise_z.isChecked())
            vp['scatter_noise_x'].setVisible(self.ui.chk_noise_x.isChecked())
            vp['scatter_l0'].setVisible(self.ui.chk_l0.isChecked())
            if 'line_avg_h' in vp:
                vp['line_avg_h'].setVisible(self.ui.chk_avg_h.isChecked())
        if 'line_apex_all' in self.ui.vp_3d:
            self.ui.vp_3d['line_apex_all'].setVisible(self.ui.chk_line_apex.isChecked())

    def toggle_ortho(self, state):
        if not hasattr(self, 'ui'): return
        vps = [self.ui.vp_main, self.ui.vp_3d] + self.ui.vp_splits + getattr(self.ui, 'vp_multi', [])
        for vp in vps:
            if state:
                vp['view'].opts['fov'] = 1 
                vp['view'].setCameraPosition(distance=2000) 
            else:
                vp['view'].opts['fov'] = 60
                vp['view'].setCameraPosition(distance=50)
            vp['view'].update()

    def toggle_export_menu(self):
        is_visible = self.ui.export_panel.isVisible()
        self.ui.export_panel.setVisible(not is_visible)
        if is_visible:
            self.ui.btn_toggle_right.setArrowType(QtCore.Qt.ArrowType.LeftArrow)
        else:
            self.ui.btn_toggle_right.setArrowType(QtCore.Qt.ArrowType.RightArrow)

    def toggle_benchmark(self):
        if self.ui.dock_bench.isVisible():
            self.ui.dock_bench.hide()
        else:
            self.ui.dock_bench.show()
            
        sz = self.size()
        self.resize(sz.width(), sz.height() + 1)
        self.resize(sz)

    def on_benchmark_visibility(self, visible):
        if visible and not self._is_working:
            self._execute_refresh(self.run_benchmark)
        self._execute_refresh(self.renderer.refresh)

    def calculate_filter_speed(self):
        if self.full_df is None: return
        self.show_loading()
        algos = ["DBSCAN", "Median", "SOR"]
        eval_layers = [l for l in self.layers if l > 0]
        if not eval_layers:
            self.hide_loading()
            return
            
        for a in algos:
            total_time = 0.0
            valid_layers = 0
            for i, lid in enumerate(eval_layers):
                self.renderer.update_progress(i, len(eval_layers))
                target, _ = Processor.prepare_layer(self.full_df, lid)
                if target is None or target.empty: continue
                t = Processor.benchmark_algorithm(target, self.params, a, 50)
                if t is not None:
                    total_time += t
                    valid_layers += 1
            if valid_layers > 0:
                self.filter_times[a] = f"{total_time / valid_layers:.2f}"
                
        self.renderer.update_progress(100, 100)
        self.hide_loading()
        if self.ui.dock_bench.isVisible():
            self.run_benchmark()

    def run_benchmark(self):
        if self.full_df is None or self.current_view_mode in ['INTRA', 'MULTI']:
            return

        split_mode = self.ui.chk_split_mode.isChecked()
        self.ui.table_bench_sma.setVisible(split_mode)

        self.ui.table_bench.setRowCount(0)
        self.ui.table_bench_sma.setRowCount(0)
        algos = ["NoFilter", "DBSCAN", "Median", "SOR"]
        sma_win = self.ui.sma_win.value()
        
        spike_thr = self.ui.spin_spike_thr.value()
        fft_cutoff = self.ui.spin_fft_cutoff.value()
        lag_step = self.ui.spin_lag_step.value()
        use_raw = self.ui.chk_raw_apexes.isChecked()

        if self.current_view_mode == 'INTER':
            lid = self.ui.cb_layer.value()
            target, _ = Processor.prepare_layer(self.full_df, lid)
            if target is None or target.empty: return
            
            for algo in algos:
                valid, _, _, _ = Processor.filter_data(target, self.params, algo)
                if valid is None or valid.empty: continue
                if use_raw:
                    apexes = Processor.calc_raw_apexes(valid)
                else:
                    apexes, _, _, _ = Processor.calc_apexes(valid, self.params)
                spikes, f_rat, lag1, _ = MetricsEvaluator.evaluate(apexes, spike_thr, fft_cutoff, lag_step)
                
                rc = self.ui.table_bench.rowCount()
                self.ui.table_bench.insertRow(rc)
                self.ui.table_bench.setItem(rc, 0, QtWidgets.QTableWidgetItem(algo))
                self.ui.table_bench.setItem(rc, 1, QtWidgets.QTableWidgetItem(str(spikes)))
                self.ui.table_bench.setItem(rc, 2, QtWidgets.QTableWidgetItem(f"{f_rat:.2f}"))
                self.ui.table_bench.setItem(rc, 3, QtWidgets.QTableWidgetItem(f"{lag1:.3f}"))
                self.ui.table_bench.setItem(rc, 4, QtWidgets.QTableWidgetItem(self.filter_times.get(algo, "-")))

                if split_mode:
                    ap_smooth = Processor.apply_smoothing(apexes.sort_values('Y') if not apexes.empty else apexes, 'SMA', sma_win)
                    spikes_s, f_rat_s, lag1_s, _ = MetricsEvaluator.evaluate(ap_smooth, spike_thr, fft_cutoff, lag_step)
                    rc_s = self.ui.table_bench_sma.rowCount()
                    self.ui.table_bench_sma.insertRow(rc_s)
                    self.ui.table_bench_sma.setItem(rc_s, 0, QtWidgets.QTableWidgetItem(algo))
                    self.ui.table_bench_sma.setItem(rc_s, 1, QtWidgets.QTableWidgetItem(str(spikes_s)))
                    self.ui.table_bench_sma.setItem(rc_s, 2, QtWidgets.QTableWidgetItem(f"{f_rat_s:.2f}"))
                    self.ui.table_bench_sma.setItem(rc_s, 3, QtWidgets.QTableWidgetItem(f"{lag1_s:.3f}"))
                    self.ui.table_bench_sma.setItem(rc_s, 4, QtWidgets.QTableWidgetItem(self.filter_times.get(algo, "-")))
                
        elif self.current_view_mode == '3D':
            min_l = self.ui.cb_layer_min.value()
            max_l = self.ui.cb_layer_max.value()
            eval_layers = [l for l in self.layers if min_l <= l <= max_l]
            
            if not eval_layers: return
            
            for algo in algos:
                tot_spikes, tot_f, tot_lag, valid_l = 0, 0.0, 0.0, 0
                tot_spikes_s, tot_f_s, tot_lag_s, valid_l_s = 0, 0.0, 0.0, 0

                for i, lid in enumerate(eval_layers):
                    self.renderer.update_progress(i, len(eval_layers))
                    target, _ = Processor.prepare_layer(self.full_df, lid)
                    if target is None or target.empty: continue
                    valid, _, _, _ = Processor.filter_data(target, self.params, algo)
                    if valid is None or valid.empty: continue
                    if use_raw:
                        apexes = Processor.calc_raw_apexes(valid)
                    else:
                        apexes, _, _, _ = Processor.calc_apexes(valid, self.params)
                    spikes, f_rat, lag1, _ = MetricsEvaluator.evaluate(apexes, spike_thr, fft_cutoff, lag_step)
                    tot_spikes += spikes
                    tot_f += f_rat
                    tot_lag += lag1
                    valid_l += 1

                    if split_mode:
                        ap_smooth = Processor.apply_smoothing(apexes.sort_values('Y') if not apexes.empty else apexes, 'SMA', sma_win)
                        spikes_s, f_rat_s, lag1_s, _ = MetricsEvaluator.evaluate(ap_smooth, spike_thr, fft_cutoff, lag_step)
                        tot_spikes_s += spikes_s
                        tot_f_s += f_rat_s
                        tot_lag_s += lag1_s
                        valid_l_s += 1
                
                self.renderer.update_progress(len(eval_layers), len(eval_layers))
                
                avg_f = tot_f / valid_l if valid_l > 0 else 0.0
                avg_lag = tot_lag / valid_l if valid_l > 0 else 0.0
                
                rc = self.ui.table_bench.rowCount()
                self.ui.table_bench.insertRow(rc)
                self.ui.table_bench.setItem(rc, 0, QtWidgets.QTableWidgetItem(algo))
                self.ui.table_bench.setItem(rc, 1, QtWidgets.QTableWidgetItem(str(tot_spikes)))
                self.ui.table_bench.setItem(rc, 2, QtWidgets.QTableWidgetItem(f"{avg_f:.2f}"))
                self.ui.table_bench.setItem(rc, 3, QtWidgets.QTableWidgetItem(f"{avg_lag:.3f}"))
                self.ui.table_bench.setItem(rc, 4, QtWidgets.QTableWidgetItem(self.filter_times.get(algo, "-")))

                if split_mode:
                    avg_f_s = tot_f_s / valid_l_s if valid_l_s > 0 else 0.0
                    avg_lag_s = tot_lag_s / valid_l_s if valid_l_s > 0 else 0.0
                    rc_s = self.ui.table_bench_sma.rowCount()
                    self.ui.table_bench_sma.insertRow(rc_s)
                    self.ui.table_bench_sma.setItem(rc_s, 0, QtWidgets.QTableWidgetItem(algo))
                    self.ui.table_bench_sma.setItem(rc_s, 1, QtWidgets.QTableWidgetItem(str(tot_spikes_s)))
                    self.ui.table_bench_sma.setItem(rc_s, 2, QtWidgets.QTableWidgetItem(f"{avg_f_s:.2f}"))
                    self.ui.table_bench_sma.setItem(rc_s, 3, QtWidgets.QTableWidgetItem(f"{avg_lag_s:.3f}"))
                    self.ui.table_bench_sma.setItem(rc_s, 4, QtWidgets.QTableWidgetItem(self.filter_times.get(algo, "-")))

    def export_filter_statistics(self):
        if self.full_df is None:
            return
            
        base_dir = r"C:\Users\Timo\Desktop\AddLAB\Apps&Scripts\API\Output_Analisys_cases\Filter Statistics From App"
        os.makedirs(base_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%d_%m_%H_%M")
        filename = f"FiltersStats_{timestamp}.csv"
        filepath = os.path.join(base_dir, filename)
        
        spike_thr = self.ui.spin_spike_thr.value()
        fft_cutoff = self.ui.spin_fft_cutoff.value()
        lag_step = self.ui.spin_lag_step.value()
        use_raw = self.ui.chk_raw_apexes.isChecked()
        
        self.show_loading()
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Layer", "Algorithm", "Spikes", "FFT Noise (%)", "Lag-1 Autocorr", "Time (ms)"])
                
                algos = ["NoFilter", "DBSCAN", "Median", "SOR"]
                global_stats = {a: {'spikes': 0, 'f_rat': 0.0, 'lag1': 0.0, 'count': 0} for a in algos}
                
                eval_layers = [l for l in self.layers if l > 0]
                for i, lid in enumerate(eval_layers):
                    self.renderer.update_progress(i, len(eval_layers))
                    target, _ = Processor.prepare_layer(self.full_df, lid)
                    if target is None or target.empty: continue
                    
                    for algo in algos:
                        valid, _, _, _ = Processor.filter_data(target, self.params, algo)
                        if valid is None or valid.empty: continue
                        if use_raw:
                            apexes = Processor.calc_raw_apexes(valid)
                        else:
                            apexes, _, _, _ = Processor.calc_apexes(valid, self.params)
                        spikes, f_rat, lag1, _ = MetricsEvaluator.evaluate(apexes, spike_thr, fft_cutoff, lag_step)
                        
                        t_str = self.filter_times.get(algo, "-")
                        writer.writerow([lid, algo, spikes, f"{f_rat:.2f}", f"{lag1:.3f}", t_str])
                        
                        global_stats[algo]['spikes'] += spikes
                        global_stats[algo]['f_rat'] += f_rat
                        global_stats[algo]['lag1'] += lag1
                        global_stats[algo]['count'] += 1
                        
                writer.writerow([])
                writer.writerow(["Global", "Algorithm", "Total Spikes", "Avg FFT Noise (%)", "Avg Lag-1 Autocorr", "Time (ms)"])
                for algo in algos:
                    st = global_stats[algo]
                    c = st['count'] if st['count'] > 0 else 1
                    t_str = self.filter_times.get(algo, "-")
                    writer.writerow(["Global", algo, st['spikes'], f"{st['f_rat']/c:.2f}", f"{st['lag1']/c:.3f}", t_str])
                    
            QtWidgets.QMessageBox.information(self, "Export", f"Статистика успешно сохранена:\n{filepath}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Ошибка при экспорте:\n{str(e)}")
        finally:
            self.hide_loading()

    def export_data(self):
        if self.full_df is None:
            return
            
        algo = self.current_algorithm
        timestamp = datetime.now().strftime("%d.%m.%Y")
        default_filename = f"ExpStats_{timestamp}_{algo}.csv"
        
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Export Data", default_filename, "CSV Files (*.csv)")
        if not filepath:
            return
            
        self.show_loading()
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Layer", "Bead Avg H", "Total Height", "Total Points", "Scan Groups", "Valid Apexes", "", "h-Bead Avg H", "H-Total Height"])
                
                h_step = self.params.get('h', 0.9)
                total_h = 0.0
                prev_h_raw = 0.0
                max_bead_error = 0.0
                
                for i, lid in enumerate(self.layers):
                    self.renderer.update_progress(i, len(self.layers))
                    curr_h_raw, _, _, _, _, stats, total_points = self.renderer._get_cached_layer_data(lid)
                    
                    if total_points > 0:
                        if lid > 0:
                            bead_thickness = (curr_h_raw - prev_h_raw) + (0.0 if lid == 1 else h_step)
                            total_h += bead_thickness
                        else:
                            bead_thickness = 0.0
                        prev_h_raw = curr_h_raw
                    else:
                        bead_thickness = 0.0
                        
                    if lid > 0 and total_points > 0:
                        bead_error = bead_thickness - h_step
                        total_error = total_h - (lid * h_step)
                        if abs(bead_error) > abs(max_bead_error):
                            max_bead_error = bead_error
                    else:
                        bead_error = 0.0
                        total_error = 0.0
                        
                    writer.writerow([
                        lid, 
                        f"{bead_thickness:.3f}", 
                        f"{total_h:.3f}", 
                        total_points, 
                        stats.get('total_groups', 0), 
                        stats.get('valid', 0),
                        "",
                        f"{bead_error:.3f}",
                        f"{total_error:.3f}"
                    ])
                    
                self.renderer.update_progress(len(self.layers), len(self.layers))
                
                writer.writerow([])
                writer.writerow(["", "", "", "", "", "", "Max Error:", f"{max_bead_error:.3f}", ""])
                writer.writerow([])
                writer.writerow(["--- Filter Parameters ---"])
                writer.writerow(["Algorithm", algo])
                for k, v in self.params.items():
                    writer.writerow([k, v])
                    
            QtWidgets.QMessageBox.information(self, "Export", f"Данные успешно сохранены:\n{filepath}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Ошибка при экспорте:\n{str(e)}")
        finally:
            self.hide_loading()

    def start_calibration(self):
        if self.full_df is None: return
        self.ui.btn_calib_start.setEnabled(False)
        self.ui.btn_calib_stop.setEnabled(True)
        self.ui.calib_progress.setVisible(True)
        self.ui.calib_progress.setValue(0)

        algo = self.current_algorithm
        iters = self.ui.spin_calib_iter.value()
        use_raw = self.ui.chk_raw_apexes.isChecked()
        spike_thr = self.ui.spin_spike_thr.value()
        fft = self.ui.spin_fft_cutoff.value()
        lag = self.ui.spin_lag_step.value()
        spike_penalty = self.ui.spin_calib_penalty.value()

        if self.ui.chk_calib_all.isChecked():
            target_layers = [l for l in self.layers if l > 0]
        else:
            min_l = self.ui.spin_calib_min.value()
            max_l = self.ui.spin_calib_max.value()
            target_layers = [l for l in self.layers if min_l <= l <= max_l]

        self.calib_thread = CalibrationThread(self.full_df, target_layers, algo, self.params, self.param_meta, iters, use_raw, spike_thr, fft, lag, spike_penalty)
        self.calib_thread.progress.connect(self.on_calib_progress)
        self.calib_thread.finished_calib.connect(self.on_calib_finished)
        self.calib_thread.start()

    def stop_calibration(self):
        if self.calib_thread is not None:
            self.calib_thread.stop()
        self.ui.btn_calib_start.setEnabled(True)
        self.ui.btn_calib_stop.setEnabled(False)
        self.ui.calib_progress.setVisible(False)

    def on_calib_progress(self, current, total):
        pct = int((current / total) * 100)
        self.ui.calib_progress.setValue(pct)

    def on_calib_finished(self, best_params, best_score):
        for k, v in best_params.items():
            if k in self.ui.inputs:
                self.ui.inputs[k].blockSignals(True)
                self.ui.inputs[k].setValue(v)
                self.params[k] = v
                self.ui.inputs[k].blockSignals(False)

        self.ui.btn_calib_start.setEnabled(True)
        self.ui.btn_calib_stop.setEnabled(False)
        self.ui.calib_progress.setVisible(False)
        self.update_params()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')
    w = VisualizerWindow()
    w.show()
    sys.exit(app.exec())