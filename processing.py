import os
import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
from scipy.fft import fft
from config import *

try:
    from sklearn.cluster import DBSCAN
    DBSCAN_AVAILABLE = True
except ImportError:
    DBSCAN_AVAILABLE = False

class MetricsEvaluator:
    @staticmethod
    def evaluate(apexes, spike_thr=1.0, fft_cutoff=0.25, lag_step=1):
        if apexes is None or apexes.empty or len(apexes) <= lag_step:
            return 0, 0.0, 0.0, pd.DataFrame()
        
        ap = apexes.sort_values('Y')
        y = ap['Y'].values
        z = ap['Z'].values
        
        dy = np.diff(y)
        dz = np.diff(z)
        grad = np.abs(dz / np.where(dy == 0, 1e-6, dy))
        
        spike_mask = grad > spike_thr
        spikes = int(np.sum(spike_mask))
        spike_df = ap.iloc[1:][spike_mask]
        
        try:
            Z_f = np.abs(fft(z - np.mean(z)))
            n = len(Z_f)
            half = n // 2
            if half > 2:
                idx = max(1, int(half * fft_cutoff))
                low_f = np.sum(Z_f[1:idx])
                high_f = np.sum(Z_f[idx:half])
                tot = low_f + high_f
                fft_ratio = (high_f / tot * 100) if tot > 0 else 0.0
            else:
                fft_ratio = 0.0
        except:
            fft_ratio = 0.0
            
        try:
            z_c = z - np.mean(z)
            num = np.sum(z_c[:-lag_step] * z_c[lag_step:])
            den = np.sum(z_c**2)
            lag1 = num / den if den > 0 else 0.0
        except:
            lag1 = 0.0
            
        return spikes, fft_ratio, lag1, spike_df

class Processor:
    @staticmethod
    def load_data(filepath, speed=10.0, step=0.5):
        if not os.path.exists(filepath):
            return None
        try:
            df = pd.read_csv(filepath)
            
            processed = []
            for lid in df['Layer'].unique():
                g = df[df['Layer'] == lid].copy()
                if 'Timestamp' in g.columns and not g['Timestamp'].isnull().all():
                    t_min = g['Timestamp'].min()
                    g['Y_Calc'] = (g['Timestamp'] - t_min) * speed
                    if 'ScanID' in g.columns:
                        g['Scan_Group'] = g['ScanID']
                    else:
                        g['Scan_Group'] = (g['PointNr'].diff() < 0).cumsum()
                else:
                    if 'ScanID' in g.columns:
                        g['Y_Calc'] = (g['ScanID'] - g['ScanID'].min()) * step
                        g['Scan_Group'] = g['ScanID']
                    else:
                        jumps = (g['PointNr'].diff() < 0).cumsum()
                        g['Y_Calc'] = jumps * step
                        g['Scan_Group'] = jumps
                processed.append(g)
            
            full_df = pd.concat(processed)
            
            l0 = full_df[full_df['Layer'] == 0]
            if not l0.empty:
                try:
                    from scipy.interpolate import NearestNDInterpolator
                    interp = NearestNDInterpolator(
                        list(zip(l0['X'], l0['Y_Calc'])), 
                        l0['Z']
                    )
                    z_base_at_point = interp(full_df['X'], full_df['Y_Calc'])
                    full_df['H_Local'] = z_base_at_point - full_df['Z']
                except Exception:
                    full_df['H_Local'] = 0.0
            else:
                full_df['H_Local'] = 0.0
            
            layer_bounds = {}
            valid_lengths = []
            
            for layer_id in full_df['Layer'].unique():
                if layer_id == 0:
                    continue
                
                group = full_df[full_df['Layer'] == layer_id]
                mask_roi = (group['X'] >= DEF_X_MIN) & (group['X'] <= DEF_X_MAX)
                roi_points = group[mask_roi]
                
                if roi_points.empty:
                    continue
                
                bead_points = roi_points[roi_points['H_Local'] > DEF_ALIGN_H]
                
                if not bead_points.empty:
                    y_start = bead_points['Y_Calc'].min()
                    y_end = bead_points['Y_Calc'].max()
                    length = y_end - y_start
                    
                    if length > 5.0:
                        layer_bounds[layer_id] = (y_start, y_end, length)
                        valid_lengths.append(length)
                        
            if valid_lengths:
                target_length = float(np.median(valid_lengths))
                full_df['Scale_Factor'] = 1.0
                full_df['Bead_Start_Y'] = 0.0
                full_df['Bead_End_Y'] = target_length
                
                for layer_id in full_df['Layer'].unique():
                    if layer_id == 0:
                        continue
                    mask = full_df['Layer'] == layer_id
                    if layer_id in layer_bounds:
                        y_s, y_e, length_raw = layer_bounds[layer_id]
                        scale_factor = target_length / length_raw if length_raw > 0 else 1.0
                        
                        full_df.loc[mask, 'Y_Calc'] = (full_df.loc[mask, 'Y_Calc'] - y_s) * scale_factor
                        full_df.loc[mask, 'Scale_Factor'] = scale_factor
                        full_df.loc[mask, 'Bead_Start_Y'] = 0.0
                        full_df.loc[mask, 'Bead_End_Y'] = target_length
                    else:
                        y_min = full_df.loc[mask, 'Y_Calc'].min()
                        y_max = full_df.loc[mask, 'Y_Calc'].max()
                        full_df.loc[mask, 'Bead_Start_Y'] = float(y_min) if pd.notnull(y_min) else 0.0
                        full_df.loc[mask, 'Bead_End_Y'] = float(y_max) if pd.notnull(y_max) else 0.0
                        
            cols_to_float32 = ['X', 'Y_Calc', 'H_Local', 'Z']
            for col in cols_to_float32:
                if col in full_df.columns:
                    full_df[col] = full_df[col].astype(np.float32)
                        
            return full_df
        except:
            return None

    @staticmethod
    def prepare_layer(df, layer_id, layer_0_id=0):
        if df is None or df.empty:
            return None, None
        target = df[df['Layer'] == layer_id].copy()
        l0 = df[df['Layer'] == layer_0_id].copy()
        return target, l0

    @staticmethod
    def _get_roi(df, params):
        if df is None or df.empty:
            return None, None

        if 'Scale_Factor' in df.columns:
            sf = float(df['Scale_Factor'].iloc[0])
            b_start = float(df['Bead_Start_Y'].iloc[0])
            b_end = float(df['Bead_End_Y'].iloc[0])
        else:
            sf = 1.0
            bead_points = df[df['H_Local'] > DEF_ALIGN_H]
            if not bead_points.empty:
                b_start = bead_points['Y_Calc'].min()
                b_end = bead_points['Y_Calc'].max()
            else:
                b_start = df['Y_Calc'].min()
                b_end = df['Y_Calc'].max()

        trim_start_y = b_start + (params['TRIM_START'] * sf)
        trim_end_y = b_end - (params['TRIM_END'] * sf)

        mask_ends = (df['Y_Calc'] < trim_start_y) | (df['Y_Calc'] > trim_end_y)
        ends_df = df[mask_ends]
        work_df = df[~mask_ends]

        if work_df.empty:
            return None, ends_df

        roi_mask = (work_df['X'] >= params['X_MIN']) & (work_df['X'] <= params['X_MAX'])
        roi = work_df[roi_mask].copy()
        
        return roi, ends_df

    @staticmethod
    def apply_algorithm(roi_valid, algorithm, params):
        valid = roi_valid
        noise_z = pd.DataFrame()

        if algorithm == 'NoFilter':
            pass

        elif algorithm == 'DBSCAN':
            if not DBSCAN_AVAILABLE:
                pass
            else:
                try:
                    coords = np.column_stack([
                        roi_valid['X'].values / params['CONN_DX'],
                        roi_valid['Y_Calc'].values / params['CONN_DY'],
                        roi_valid['H_Local'].values / params['CONN_DZ']
                    ]).astype(np.float32)
                    
                    db = DBSCAN(eps=params['EPS'], min_samples=int(params['MIN_SAMPLES'])).fit(coords)
                    
                    temp_df = roi_valid.copy()
                    temp_df['Cluster'] = db.labels_
                    valid_clusters = temp_df['Cluster'].value_counts()[lambda x: x >= params['MIN_CLUSTER_SIZE']].index
                    mask_valid = temp_df['Cluster'].isin(valid_clusters)
                    
                    valid = temp_df[mask_valid].drop(columns=['Cluster'])
                    noise_z = temp_df[~mask_valid].drop(columns=['Cluster'])
                except:
                    valid = roi_valid

        elif algorithm == 'Median':
            win = int(params['MEDIAN_WINDOW'])
            thresh = params['MEDIAN_THRESHOLD']
            
            roi_sorted = roi_valid.sort_values(['Scan_Group', 'X'])
            sg_vals = roi_sorted['Scan_Group'].values
            h_vals = roi_sorted['H_Local'].values
            
            splits = np.nonzero(sg_vals[1:] != sg_vals[:-1])[0] + 1
            h_groups = np.split(h_vals, splits)
            
            clean_masks = []
            for hg in h_groups:
                if len(hg) < win:
                    clean_masks.append(np.zeros(len(hg), dtype=bool))
                    continue
                med = median_filter(hg, size=win, mode='nearest')
                diff = np.abs(hg - med)
                clean_masks.append(diff < thresh)
                
            is_clean = np.concatenate(clean_masks)
            valid = roi_sorted[is_clean]
            noise_z = roi_sorted[~is_clean]

        elif algorithm == 'SOR':
            coords = np.column_stack([
                roi_valid['X'].values,
                roi_valid['Y_Calc'].values,
                roi_valid['H_Local'].values
            ]).astype(np.float32)
            tree = cKDTree(coords)
            k = int(params['SOR_K']) + 1
            distances, _ = tree.query(coords, k=k)
            mean_dist = np.mean(distances[:, 1:], axis=1)
            global_mean = np.mean(mean_dist)
            global_std = np.std(mean_dist)
            threshold = global_mean + params['SOR_MULT'] * global_std
            mask = mean_dist <= threshold
            valid = roi_valid[mask]
            noise_z = roi_valid[~mask]

        return valid, noise_z

    @staticmethod
    def benchmark_algorithm(df, params, algorithm, iterations=50):
        roi, _ = Processor._get_roi(df, params)
        if roi is None or roi.empty:
            return None
            
        z_thresh = np.percentile(roi['Z'].values, params['PERC_MIN']) + params['DELTA_H']
        roi_valid = roi[roi['Z'] <= z_thresh]
        if roi_valid.empty:
            return None
            
        start_t = time.perf_counter()
        for _ in range(iterations):
            Processor.apply_algorithm(roi_valid, algorithm, params)
        end_t = time.perf_counter()
        
        return ((end_t - start_t) / iterations) * 1000.0

    @staticmethod
    def filter_data(df, params, algorithm='NoFilter'):
        roi, ends_df = Processor._get_roi(df, params)
        if roi is None or roi.empty:
            return None, None, None, ends_df

        z_thresh = np.percentile(roi['Z'].values, params['PERC_MIN']) + params['DELTA_H']
        roi_valid = roi[roi['Z'] <= z_thresh]
        noise_z_perc = roi[roi['Z'] > z_thresh]

        if roi_valid.empty:
            return None, noise_z_perc, None, ends_df

        valid, algo_noise_z = Processor.apply_algorithm(roi_valid, algorithm, params)
        noise_z = pd.concat([noise_z_perc, algo_noise_z]) if not algo_noise_z.empty else noise_z_perc

        if valid.empty:
            return None, noise_z, None, ends_df

        med_x = valid['X'].median()
        x_tol = params['X_TOL']
        mask_x = (valid['X'] >= (med_x - x_tol)) & (valid['X'] <= (med_x + x_tol))
        
        final_valid = valid[mask_x]
        noise_x = valid[~mask_x]

        return final_valid, noise_z, noise_x, ends_df

    @staticmethod
    def calc_raw_apexes(df):
        if df is None or df.empty:
            return pd.DataFrame()

        apexes = []
        for sg, group in df.groupby('Scan_Group'):
            if group.empty:
                continue
            
            max_idx = group['H_Local'].idxmax()
            peak = group.loc[max_idx]
            
            apexes.append({
                'X': np.float32(peak['X']),
                'Y': np.float32(peak['Y_Calc']),
                'Z': np.float32(peak['H_Local'])
            })
            
        return pd.DataFrame(apexes)

    @staticmethod
    def calc_apexes(df, params):
        stats = {
            'total_groups': 0, 'low_pts': 0, 'narrow': 0, 
            'fit_error': 0, 'x_out_roi': 0, 'bad_curvature': 0, 
            'spatial_outlier': 0, 'valid': 0
        }

        if df is None or df.empty:
            return pd.DataFrame(), pd.DataFrame(), 0.0, stats

        df_sorted = df.sort_values('Scan_Group')
        x_vals = df_sorted['X'].values
        y_vals = df_sorted['Y_Calc'].values
        z_vals = df_sorted['H_Local'].values
        sg_vals = df_sorted['Scan_Group'].values
        
        splits = np.nonzero(sg_vals[1:] != sg_vals[:-1])[0] + 1
        x_groups = np.split(x_vals, splits)
        y_groups = np.split(y_vals, splits)
        z_groups = np.split(z_vals, splits)

        stats['total_groups'] = len(x_groups)
        temp_apexes = []
        rejects = []

        for xg, yg, zg in zip(x_groups, y_groups, z_groups):
            if len(xg) < params['MIN_PTS']: 
                stats['low_pts'] += 1
                continue
            
            if xg.max() - xg.min() < params['MIN_WIDTH']: 
                stats['narrow'] += 1
                continue

            try:
                z_max_local = zg.max()
                cap_mask = zg >= (z_max_local - 1.0) 
                
                if np.sum(cap_mask) >= 3:
                    xg_fit, zg_fit = xg[cap_mask], zg[cap_mask]
                else:
                    xg_fit, zg_fit = xg, zg
                    
                coeffs = np.polyfit(xg_fit, zg_fit, 2)
                a, b, c = coeffs
                
                if abs(a) > 1e-6:
                    vx = -b / (2*a)
                else:
                    vx = xg.mean()

                if not (params['X_MIN'] <= vx <= params['X_MAX']): 
                    stats['x_out_roi'] += 1
                    continue

                vz = a*(vx**2) + b*vx + c
                
                if abs(vz - z_max_local) > 0.3:
                    vz = z_max_local

                vy = yg.mean()
                rec = {'X': np.float32(vx), 'Y': np.float32(vy), 'Z': np.float32(vz), 'A': np.float32(a)}

                if a >= -params['MIN_CURV']:
                    rejects.append(rec)
                    stats['bad_curvature'] += 1
                else:
                    temp_apexes.append(rec)

            except:
                stats['fit_error'] += 1

        med_x = 0.0
        final_apexes = pd.DataFrame()
        
        if temp_apexes:
            df_temp = pd.DataFrame(temp_apexes)
            med_x = df_temp['X'].median()
            limit = params['APEX_TOL']
            mask = (df_temp['X'] >= med_x - limit) & (df_temp['X'] <= med_x + limit)
            final_apexes = df_temp[mask]
            stats['spatial_outlier'] = len(df_temp[~mask])
            stats['valid'] = len(final_apexes)

        res_rej = pd.DataFrame(rejects)
        return final_apexes, res_rej, med_x, stats

    @staticmethod
    def apply_smoothing(df, f_type, param_window):
        if df.empty or f_type == 'None':
            return df
        res = df.copy()
        if f_type == 'SMA':
            res['Z'] = res['Z'].rolling(param_window, center=True, min_periods=1).mean().astype(np.float32)
        return res

class ControlLogic:
    @staticmethod
    def compute_segments(apex_df, segment_length_mm, kp, deadband_mm, clip_start, clip_end_y, override_target_height=None):
        if apex_df is None or apex_df.empty:
            return None, None, None, 0.0

        if clip_start >= clip_end_y:
             return None, None, None, 0.0

        clipped_df = apex_df[(apex_df['Y'] >= clip_start) & (apex_df['Y'] <= clip_end_y)]
        if clipped_df.empty:
            return None, None, None, 0.0

        if override_target_height is not None:
            dynamic_target_height = override_target_height
        else:
            dynamic_target_height = clipped_df['Z'].median()

        bins = np.arange(clip_start, clip_end_y + segment_length_mm, segment_length_mm)
        if len(bins) > 0 and bins[-1] > clip_end_y + 0.001: 
             bins[-1] = clip_end_y
        if len(bins) < 2:
             return None, None, None, 0.0

        y_apex = clipped_df['Y'].values
        z_apex = clipped_df['Z'].values
        bin_indices = np.digitize(y_apex, bins)

        seg_y, seg_z, seg_speed = [], [], []

        for i in range(1, len(bins)):
            mask = bin_indices == i
            points_in_bin = z_apex[mask]

            if len(points_in_bin) > 0:
                median_h = np.median(points_in_bin)
            else:
                median_h = dynamic_target_height

            error = median_h - dynamic_target_height
            speed_correction = 0.0 if abs(error) < deadband_mm else error * kp
            speed_pct = max(60.0, min(100.0 + speed_correction, 140.0))

            seg_y.append(bins[i-1]) 
            seg_z.append(median_h)
            seg_speed.append(speed_pct)

        seg_y.append(bins[-1])
        return np.array(seg_y, dtype=np.float32), np.array(seg_z, dtype=np.float32), np.array(seg_speed, dtype=np.float32), float(dynamic_target_height)