import numpy as np
import pyqtgraph as pg
import pandas as pd
import pyqtgraph.opengl as gl
from PyQt6 import QtGui, QtWidgets
from processing import Processor, ControlLogic
from processing import MetricsEvaluator

class RenderLogic:
    def __init__(self, mw):
        self.mw = mw
        self.last_pct = -1

    def update_progress(self, current, total):
        if total > 0 and hasattr(self.mw.ui, 'progress_bar'):
            pct = int((current / total) * 100)
            if pct != self.last_pct:
                self.last_pct = pct
                self.mw.ui.progress_bar.setValue(pct)
                QtWidgets.QApplication.processEvents()

    def update_viewport(self, vp, target, l0, valid, nz, nx, apexes, colors, bead_h, total_h, total_points, stats, lid, layer_offset, fallback_used=False, spikes_df=None, algo_name=None):
        if 'warning_label' in vp:
            vp['warning_label'].setVisible(fallback_used)

        if not l0.empty:
            pos_l0 = np.vstack([l0['X'], l0['Y_Calc'], np.zeros(len(l0))]).T.astype(np.float32)
            vp['scatter_l0'].setData(pos=pos_l0)
        else:
            vp['scatter_l0'].setData(pos=np.zeros((0,3), dtype=np.float32))

        if valid is not None and not valid.empty:
            pos_v = np.vstack([valid['X'], valid['Y_Calc'], valid['H_Local'] + layer_offset]).T.astype(np.float32)
            vp['scatter_valid'].setData(pos=pos_v, color=colors)
            
            y_min = valid['Y_Calc'].min()
            y_max = valid['Y_Calc'].max()
            x_mid = valid['X'].median()
            z_val = total_h
            pts = []
            cur_y = y_min
            while cur_y < y_max:
                next_y = min(cur_y + 2.0, y_max)
                pts.append([x_mid, cur_y, z_val])
                pts.append([x_mid, next_y, z_val])
                cur_y = next_y + 2.0
            if pts:
                vp['line_avg_h'].setData(pos=np.array(pts, dtype=np.float32), mode='lines')
            else:
                vp['line_avg_h'].setData(pos=np.zeros((0,3), dtype=np.float32))
        else:
            vp['scatter_valid'].setData(pos=np.zeros((0,3), dtype=np.float32))
            vp['line_avg_h'].setData(pos=np.zeros((0,3), dtype=np.float32))

        if nz is not None and not nz.empty:
            pos_nz = np.vstack([nz['X'], nz['Y_Calc'], nz['H_Local'] + layer_offset]).T.astype(np.float32)
            vp['scatter_noise_z'].setData(pos=pos_nz)
        else:
            vp['scatter_noise_z'].setData(pos=np.zeros((0,3), dtype=np.float32))

        if nx is not None and not nx.empty:
            pos_nx = np.vstack([nx['X'], nx['Y_Calc'], nx['H_Local'] + layer_offset]).T.astype(np.float32)
            vp['scatter_noise_x'].setData(pos=pos_nx)
        else:
            vp['scatter_noise_x'].setData(pos=np.zeros((0,3), dtype=np.float32))

        if not apexes.empty:
            pos_ap = np.vstack([apexes['X'], apexes['Y'], apexes['Z'] + layer_offset]).T.astype(np.float32)
            vp['scatter_apex'].setData(pos=pos_ap)
            pos_line = np.vstack([apexes['X'], apexes['Y'], apexes['Z'] + layer_offset]).T.astype(np.float32)
            vp['line_apex'].setData(pos=pos_line)
        else:
            vp['scatter_apex'].setData(pos=np.zeros((0,3), dtype=np.float32))
            vp['line_apex'].setData(pos=np.zeros((0,3), dtype=np.float32))

        if spikes_df is not None and not spikes_df.empty and self.mw.ui.dock_bench.isVisible():
            pos_spikes = np.vstack([spikes_df['X'], spikes_df['Y'], spikes_df['Z'] + layer_offset]).T.astype(np.float32)
            vp['scatter_spikes'].setData(pos=pos_spikes)
            vp['scatter_spikes'].setVisible(True)
        else:
            if 'scatter_spikes' in vp:
                vp['scatter_spikes'].setData(pos=np.zeros((0,3), dtype=np.float32))
                vp['scatter_spikes'].setVisible(False)

        algo_str = algo_name if algo_name else self.mw.current_algorithm

        info_text = (
            f"LAYER {lid} STATS\n"
            f"-----------------\n"
            f"Algorithm:    {algo_str}\n"
            f"Total Points: {total_points}\n"
            f"Total Layers: {lid}\n"
            f"Scan Groups:  {stats['total_groups']}\n"
            f"Bead Avg H:   {bead_h:.2f} mm\n"
            f"Total Height: {total_h:.2f} mm\n"
            f"Valid Apexes: {stats['valid']}\n\n"
            f"FILTERED GROUPS:\n"
            f"- Out of ROI:    {stats['x_out_roi']}\n"
            f"- Too Few Pts:   {stats['low_pts']}\n"
            f"- Too Narrow:    {stats['narrow']}\n"
            f"- Fit Error:     {stats['fit_error']}\n"
            f"- Bad Curvature: {stats['bad_curvature']}\n"
            f"- Spatial Outlr: {stats['spatial_outlier']}"
        )
        vp['info_label'].setText(info_text)

    def _get_cached_layer_data(self, lid):
        if lid in self.mw.layer_cache:
            return self.mw.layer_cache[lid]
            
        target, _ = Processor.prepare_layer(self.mw.full_df, lid)
        if target is None or target.empty:
            data = (0.0, None, None, None, pd.DataFrame(), {}, 0)
            self.mw.layer_cache[lid] = data
            return data
            
        total_points = len(target)
        valid, nz, nx, _ = Processor.filter_data(target, self.mw.params, self.mw.current_algorithm)
        
        use_raw = self.mw.ui.chk_raw_apexes.isChecked() if hasattr(self.mw.ui, 'chk_raw_apexes') else False
        
        h_prev = 0.0
        apexes = pd.DataFrame()
        stats = {'total_groups': 0, 'low_pts': 0, 'narrow': 0, 'fit_error': 0, 'x_out_roi': 0, 'bad_curvature': 0, 'spatial_outlier': 0, 'valid': 0}
        
        if valid is not None and not valid.empty:
            if use_raw:
                apexes = Processor.calc_raw_apexes(valid)
                stats['total_groups'] = len(valid['Scan_Group'].unique())
                stats['valid'] = len(apexes)
            else:
                apexes, _, _, stats = Processor.calc_apexes(valid, self.mw.params)
                
            if not apexes.empty:
                h_prev = apexes['Z'].mean()
            else:
                h_prev = valid['H_Local'].mean()
        else:
            h_prev = target['H_Local'].mean() if not target.empty else 0.0
                
        data = (h_prev, valid, nz, nx, apexes, stats, total_points)
        self.mw.layer_cache[lid] = data
        return data

    def refresh(self):
        if self.mw.full_df is None: return
        
        current_params_tuple = tuple(self.mw.params.items())
        if self.mw.last_params != current_params_tuple:
            self.mw.layer_cache.clear()
            self.mw.last_params = current_params_tuple
            
        lid = self.mw.ui.cb_layer.value()
        
        layer_changed = getattr(self.mw, 'last_layer', None) != lid
        self.mw.last_layer = lid
        
        algo_changed = getattr(self.mw, 'last_algo_cam', None) != self.mw.current_algorithm
        self.mw.last_algo_cam = self.mw.current_algorithm
        
        should_recenter = layer_changed or algo_changed

        target, l0 = Processor.prepare_layer(self.mw.full_df, lid)
        if target is None: return

        h_step = self.mw.params.get('h', 0.9)
        prev_total_h = 0.0
        prev_h_raw = 0.0
        
        for l in sorted(self.mw.layers):
            if l <= 0 or l >= lid: continue
            curr_h_raw, _, _, _, _, _, t_pts = self._get_cached_layer_data(l)
            if t_pts > 0:
                layer_thickness = (curr_h_raw - prev_h_raw) + (0.0 if l == 1 else h_step)
                prev_total_h += layer_thickness
                prev_h_raw = curr_h_raw

        curr_h_raw, valid, nz, nx, apexes, stats, total_points = self._get_cached_layer_data(lid)
        
        bead_thickness = 0.0
        true_total_h = prev_total_h
        layer_offset = 0.0
        colors = None
        fallback_used = False
        
        if valid is not None and not valid.empty:
            c_val = valid['H_Local'].values
            c_norm = (c_val - c_val.min()) / (c_val.max() - c_val.min() + 1e-6)
            pos = np.array([0.0, 0.5, 1.0])
            color = np.array([
                [0, 0, 255, 255],
                [0, 255, 0, 255],
                [255, 0, 0, 255]
            ], dtype=np.ubyte)
            cmap = pg.ColorMap(pos, color)
            colors = cmap.map(c_norm, mode='float').astype(np.float32)
            
            if apexes.empty:
                fallback_used = True
        else:
            fallback_used = True
            
        if total_points > 0:
            if lid > 0:
                bead_thickness = (curr_h_raw - prev_h_raw) + (0.0 if lid == 1 else h_step)
                true_total_h += bead_thickness
                layer_offset = true_total_h - curr_h_raw
            else:
                bead_thickness = 0.0
                true_total_h = 0.0
                layer_offset = 0.0

        ap_sorted = apexes.sort_values('Y') if not apexes.empty else apexes
        
        spike_thr = self.mw.ui.spin_spike_thr.value()
        fft_cutoff = self.mw.ui.spin_fft_cutoff.value()
        lag_step = self.mw.ui.spin_lag_step.value()

        spikes_df_orig = pd.DataFrame()
        if not ap_sorted.empty:
            _, _, _, spikes_df_orig = MetricsEvaluator.evaluate(ap_sorted, spike_thr, fft_cutoff, lag_step)

        if not self.mw.ui.chk_split_mode.isChecked():
            self.update_viewport(self.mw.ui.vp_main, target, l0, valid, nz, nx, ap_sorted, colors, bead_thickness, true_total_h, total_points, stats, lid, layer_offset, fallback_used, spikes_df_orig)
        else:
            configs = [
                ('Original', 'None', 0),
                ('SMA Filter', 'SMA', self.mw.ui.sma_win.value())
            ]
            
            for i, (title, f_type, win) in enumerate(configs):
                vp = self.mw.ui.vp_splits[i]
                ap_smooth = Processor.apply_smoothing(ap_sorted, f_type, win)
                spikes_df_smooth = pd.DataFrame()
                if not ap_smooth.empty:
                    _, _, _, spikes_df_smooth = MetricsEvaluator.evaluate(ap_smooth, spike_thr, fft_cutoff, lag_step)
                self.update_viewport(vp, target, l0, valid, nz, nx, ap_smooth, colors, bead_thickness, true_total_h, total_points, stats, lid, layer_offset, fallback_used, spikes_df_smooth)
                vp['title_label'].setText(title)
        
        self.mw.update_visibility()

        if should_recenter and target is not None and not target.empty:
            cx = target['X'].mean()
            cy = target['Y_Calc'].mean()
            cz = layer_offset + curr_h_raw
            for vp in [self.mw.ui.vp_main] + self.mw.ui.vp_splits:
                vp['view'].opts['center'] = QtGui.QVector3D(cx, cy, cz)
                vp['view'].update()

    def clear_intra_plots(self):
        self.mw.ui.curve_apexes_2d.clear()
        self.mw.ui.curve_target_2d.clear()
        self.mw.ui.curve_segments_2d.clear()
        self.mw.ui.curve_speed_base.clear()
        self.mw.ui.curve_speed_step.clear()
        self.mw.ui.intra_info_label.setText("")

    def update_intra_plots(self, apex_df, seg_y, seg_z, seg_speed, target_h):
        if not apex_df.empty:
            self.mw.ui.curve_apexes_2d.setData(apex_df['Y'].values, apex_df['Z'].values)
            x_max = apex_df['Y'].max()
            self.mw.ui.curve_target_2d.setData([0, x_max], [target_h, target_h])
            self.mw.ui.curve_speed_base.setData([0, x_max], [100, 100])
        else:
            self.mw.ui.curve_apexes_2d.clear()
            self.mw.ui.curve_target_2d.clear()
            self.mw.ui.curve_speed_base.clear()

        info_text = "SEGMENT DEVIATIONS\n------------------\n"

        if seg_y is not None and len(seg_y) > 0 and seg_z is not None:
            self.mw.ui.curve_segments_2d.setData(seg_y, seg_z, stepMode=True)
            self.mw.ui.curve_speed_step.setData(seg_y, seg_speed, stepMode=True)
            
            for i in range(len(seg_z)):
                diff = seg_z[i] - target_h
                info_text += f"Seg {i+1}: {diff:+.2f} mm\n"
        else:
            self.mw.ui.curve_segments_2d.clear()
            self.mw.ui.curve_speed_step.clear()
            info_text += "No segments available"
            
        self.mw.ui.intra_info_label.setText(info_text.strip())

        if target_h is not None and target_h != 0.0:
            yr = self.mw.ui.plot_height.viewRange()[1]
            current_span = yr[1] - yr[0]
            
            if current_span <= 0.1:
                current_span = 30.0
                
            self.mw.ui.plot_height.setYRange(target_h - current_span / 2, target_h + current_span / 2, padding=0)

        self.mw.update_intra_visibility()

    def refresh_intra(self):
        if self.mw.full_df is None: return
        lid = self.mw.ui.cb_layer.value()

        h_step = self.mw.params.get('h', 0.9)
        total_h = 0.0
        prev_h_raw = 0.0
        
        for l in sorted(self.mw.layers):
            if l <= 0 or l >= lid: continue
            curr_h_raw, _, _, _, _, _, t_pts = self._get_cached_layer_data(l)
            if t_pts > 0:
                total_h += (curr_h_raw - prev_h_raw) + (0.0 if l == 1 else h_step)
                prev_h_raw = curr_h_raw
                
        curr_h_raw, valid, nz, nx, apexes, stats, total_points = self._get_cached_layer_data(lid)
        
        if valid is None or valid.empty or apexes.empty:
            self.clear_intra_plots()
            return
            
        if lid > 0:
            layer_thickness = (curr_h_raw - prev_h_raw) + (0.0 if lid == 1 else h_step)
            total_h += layer_thickness
            layer_offset = total_h - curr_h_raw
        else:
            layer_thickness = 0.0
            total_h = 0.0
            layer_offset = 0.0

        ap_sorted = apexes.sort_values('Y')
        if self.mw.ui.chk_split_mode.isChecked():
            ap_sorted = Processor.apply_smoothing(ap_sorted, 'SMA', self.mw.ui.sma_win.value())
            
        ap_sorted = ap_sorted.copy()
        ap_sorted['Z'] = ap_sorted['Z'] + layer_offset
            
        override_h = None
        if self.mw.ui.chk_intra_given.isChecked():
            override_h = lid * self.mw.ui.spin_given_h.value()
            
        seg_y, seg_z, seg_speed, target_h = ControlLogic.compute_segments(
            ap_sorted, 
            self.mw.ui.spin_seg_len.value(), 
            self.mw.ui.spin_kp.value(), 
            self.mw.ui.spin_deadband.value(), 
            self.mw.ui.spin_clip_start.value(), 
            self.mw.ui.spin_clip_end.value(),
            override_target_height=override_h
        )
        
        self.update_intra_plots(ap_sorted, seg_y, seg_z, seg_speed, target_h)

    def refresh_3d(self):
        if self.mw.full_df is None: return

        current_params_tuple = tuple(self.mw.params.items())
        if self.mw.last_params != current_params_tuple:
            self.mw.layer_cache.clear()
            self.mw.last_params = current_params_tuple
            
        all_v_pos, all_v_colors = [], []
        all_nz_pos, all_nx_pos = [], []
        all_ap_pos, all_ap_segs = [], []
        
        total_height_to_max = 0.0
        total_points_disp = 0
        valid_apexes_count = 0
        displayed_layers = 0
        layers_with_fallback = 0
        stats_sum = {'total_groups': 0, 'low_pts': 0, 'narrow': 0, 'fit_error': 0, 'x_out_roi': 0, 'bad_curvature': 0, 'spatial_outlier': 0, 'valid': 0}

        _, l0 = Processor.prepare_layer(self.mw.full_df, 0)
        if not l0.empty:
            pos_l0 = np.vstack([l0['X'], l0['Y_Calc'], np.zeros(len(l0))]).T.astype(np.float32)
            self.mw.ui.vp_3d['scatter_l0'].setData(pos=pos_l0)
        else:
            self.mw.ui.vp_3d['scatter_l0'].setData(pos=np.zeros((0,3), dtype=np.float32))

        min_l = self.mw.ui.cb_layer_min.value()
        max_l = self.mw.ui.cb_layer_max.value()
        layers_count = len(self.mw.layers)
        
        spike_thr = self.mw.ui.spin_spike_thr.value()
        fft_cutoff = self.mw.ui.spin_fft_cutoff.value()
        lag_step = self.mw.ui.spin_lag_step.value()

        h_step = self.mw.params.get('h', 0.9)
        total_h = 0.0
        prev_h_raw = 0.0

        for i, lid in enumerate(self.mw.layers):
            if lid <= 0: continue
            self.update_progress(i, layers_count)
            
            curr_h_raw, valid, nz, nx, apexes, stats, total_points = self._get_cached_layer_data(lid)

            if total_points == 0:
                continue
                
            layer_thickness = (curr_h_raw - prev_h_raw) + (0.0 if lid == 1 else h_step)
            total_h += layer_thickness
            layer_offset = total_h - curr_h_raw
            prev_h_raw = curr_h_raw

            if lid <= max_l:
                total_height_to_max = total_h

            if min_l <= lid <= max_l:
                total_points_disp += total_points
                displayed_layers += 1

                if valid is not None and not valid.empty:
                    if apexes.empty:
                        layers_with_fallback += 1
                        
                    for k in stats_sum: stats_sum[k] += stats[k]
                    valid_apexes_count += stats['valid']

                    pos_v = np.vstack([valid['X'], valid['Y_Calc'], valid['H_Local'] + layer_offset]).T.astype(np.float32)
                    all_v_pos.append(pos_v)
                    
                    c_val = valid['H_Local'].values
                    c_norm = (c_val - c_val.min()) / (c_val.max() - c_val.min() + 1e-6)
                    cmap = pg.ColorMap(np.array([0.0, 0.5, 1.0]), np.array([[0, 0, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255]], dtype=np.ubyte))
                    colors = cmap.map(c_norm, mode='float').astype(np.float32)
                    all_v_colors.append(colors)
                    
                    ap_sorted = apexes.sort_values('Y') if not apexes.empty else apexes
                    if self.mw.ui.chk_split_mode.isChecked() and not ap_sorted.empty:
                        ap_sorted = Processor.apply_smoothing(ap_sorted, 'SMA', self.mw.ui.sma_win.value())
                        
                    if not ap_sorted.empty:
                        pts = np.vstack([ap_sorted['X'], ap_sorted['Y'], ap_sorted['Z'] + layer_offset]).T.astype(np.float32)
                        all_ap_pos.append(pts)
                        
                        if len(pts) >= 2:
                            segs = np.empty((2 * len(pts) - 2, 3), dtype=np.float32)
                            segs[0::2] = pts[:-1]
                            segs[1::2] = pts[1:]
                            all_ap_segs.append(segs)

                if nz is not None and not nz.empty:
                    pos_nz = np.vstack([nz['X'], nz['Y_Calc'], nz['H_Local'] + layer_offset]).T.astype(np.float32)
                    all_nz_pos.append(pos_nz)

                if nx is not None and not nx.empty:
                    pos_nx = np.vstack([nx['X'], nx['Y_Calc'], nx['H_Local'] + layer_offset]).T.astype(np.float32)
                    all_nx_pos.append(pos_nx)

        self.update_progress(layers_count, layers_count)

        def safe_concat(lst, cols=3): return np.vstack(lst).astype(np.float32) if lst else np.zeros((0, cols), dtype=np.float32)

        v_pos_arr = safe_concat(all_v_pos)
        v_col_arr = safe_concat(all_v_colors, cols=4)
        if len(v_pos_arr) > 0:
            self.mw.ui.vp_3d['scatter_valid'].setData(pos=v_pos_arr, color=v_col_arr)
            
            cx = v_pos_arr[:, 0].mean()
            cy = v_pos_arr[:, 1].mean()
            cz = v_pos_arr[:, 2].mean()
            self.mw.ui.vp_3d['view'].opts['center'] = QtGui.QVector3D(cx, cy, cz)
            self.mw.ui.vp_3d['view'].update()
        else:
            self.mw.ui.vp_3d['scatter_valid'].setData(pos=np.zeros((0,3), dtype=np.float32))

        self.mw.ui.vp_3d['scatter_noise_z'].setData(pos=safe_concat(all_nz_pos))
        self.mw.ui.vp_3d['scatter_noise_x'].setData(pos=safe_concat(all_nx_pos))

        ap_pos_arr = safe_concat(all_ap_pos)
        if len(ap_pos_arr) > 0:
            self.mw.ui.vp_3d['scatter_apex'].setData(pos=ap_pos_arr)
        else:
            self.mw.ui.vp_3d['scatter_apex'].setData(pos=np.zeros((0,3), dtype=np.float32))

        if all_ap_segs:
            all_segs_arr = safe_concat(all_ap_segs)
            self.mw.ui.vp_3d['line_apex_all'].setData(pos=all_segs_arr, mode='lines')
        else:
            self.mw.ui.vp_3d['line_apex_all'].setData(pos=np.zeros((0,3), dtype=np.float32), mode='lines')

        info_text = (
            f"3D FULL VIEW STATS\n"
            f"-----------------\n"
            f"Algorithm:    {self.mw.current_algorithm}\n"
            f"Total Points: {total_points_disp}\n"
            f"Total Layers: {displayed_layers}\n"
            f"Fallback Lyrs: {layers_with_fallback}\n"
            f"Total Height: {total_height_to_max:.2f} mm\n"
            f"Valid Apexes: {valid_apexes_count}\n\n"
            f"FILTERED GROUPS (Layers {min_l}-{max_l}):\n"
            f"- Out of ROI:    {stats_sum['x_out_roi']}\n"
            f"- Too Few Pts:   {stats_sum['low_pts']}\n"
            f"- Too Narrow:    {stats_sum['narrow']}\n"
            f"- Fit Error:     {stats_sum['fit_error']}\n"
            f"- Bad Curvature: {stats_sum['bad_curvature']}\n"
            f"- Spatial Outlr: {stats_sum['spatial_outlier']}"
        )
        self.mw.ui.vp_3d['info_label'].setText(info_text)
        
        self.mw.update_visibility()

    def refresh_multi(self):
        if self.mw.full_df is None: return
        
        current_params_tuple = tuple(self.mw.params.items())
        if self.mw.last_params != current_params_tuple:
            self.mw.layer_cache.clear()
            self.mw.last_params = current_params_tuple
            
        lid = self.mw.ui.cb_layer.value()
        target, l0 = Processor.prepare_layer(self.mw.full_df, lid)
        if target is None: return

        h_step = self.mw.params.get('h', 0.9)
        prev_total_h = 0.0
        prev_h_raw = 0.0
        
        layers_count = len(self.mw.layers)
        
        for i, l in enumerate(self.mw.layers):
            if l <= 0 or l >= lid:
                self.update_progress(i, layers_count)
                continue
            curr_h_raw, _, _, _, _, _, t_pts = self._get_cached_layer_data(l)
            if t_pts > 0:
                layer_thickness = (curr_h_raw - prev_h_raw) + (0.0 if l == 1 else h_step)
                prev_total_h += layer_thickness
                prev_h_raw = curr_h_raw
            self.update_progress(i, layers_count)
            
        self.update_progress(layers_count, layers_count)

        algos = ["NoFilter", "DBSCAN", "Median", "SOR"]
        use_raw = self.mw.ui.chk_raw_apexes.isChecked() if hasattr(self.mw.ui, 'chk_raw_apexes') else False
        spike_thr = self.mw.ui.spin_spike_thr.value()
        fft_cutoff = self.mw.ui.spin_fft_cutoff.value()
        lag_step = self.mw.ui.spin_lag_step.value()
        
        for i, algo in enumerate(algos):
            self.update_progress(i, len(algos))
            vp = self.mw.ui.vp_multi[i]
            vp['title_label'].setText(f"{algo} Filter")
            
            valid, nz, nx, _ = Processor.filter_data(target, self.mw.params, algo)
            
            apexes = pd.DataFrame()
            stats = {'total_groups': 0, 'low_pts': 0, 'narrow': 0, 'fit_error': 0, 'x_out_roi': 0, 'bad_curvature': 0, 'spatial_outlier': 0, 'valid': 0}
            fallback_used = False
            algo_curr_h_raw = 0.0
            colors = None
            
            if valid is not None and not valid.empty:
                if use_raw:
                    apexes = Processor.calc_raw_apexes(valid)
                    stats['total_groups'] = len(valid['Scan_Group'].unique())
                    stats['valid'] = len(apexes)
                else:
                    apexes, _, _, stats = Processor.calc_apexes(valid, self.mw.params)
                
                c_val = valid['H_Local'].values
                c_norm = (c_val - c_val.min()) / (c_val.max() - c_val.min() + 1e-6)
                cmap = pg.ColorMap(np.array([0.0, 0.5, 1.0]), np.array([[0, 0, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255]], dtype=np.ubyte))
                colors = cmap.map(c_norm, mode='float').astype(np.float32)
                
                if not apexes.empty:
                    algo_curr_h_raw = apexes['Z'].mean()
                else:
                    algo_curr_h_raw = valid['H_Local'].mean()
                    fallback_used = True
            else:
                algo_curr_h_raw = target['H_Local'].mean() if not target.empty else 0.0
                fallback_used = True

            if lid > 0:
                algo_bead_thickness = (algo_curr_h_raw - prev_h_raw) + (0.0 if lid == 1 else h_step)
                algo_total_h = prev_total_h + algo_bead_thickness
                layer_offset = algo_total_h - algo_curr_h_raw
            else:
                algo_bead_thickness = 0.0
                algo_total_h = 0.0
                layer_offset = 0.0

            ap_sorted = apexes.sort_values('Y') if not apexes.empty else apexes
            
            spikes_df = pd.DataFrame()
            if not ap_sorted.empty:
                _, _, _, spikes_df = MetricsEvaluator.evaluate(ap_sorted, spike_thr, fft_cutoff, lag_step)

            self.update_viewport(vp, target, l0, valid, nz, nx, ap_sorted, colors, algo_bead_thickness, algo_total_h, len(target), stats, lid, layer_offset, fallback_used, spikes_df, algo_name=algo)
            
        if target is not None and not target.empty:
            cx = target['X'].mean()
            cy = target['Y_Calc'].mean()
            cz = prev_total_h
            for vp in self.mw.ui.vp_multi:
                vp['view'].opts['center'] = QtGui.QVector3D(cx, cy, cz)
                vp['view'].update()
            
        self.update_progress(len(algos), len(algos))
        self.mw.update_visibility()