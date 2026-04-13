import sys
import os
import time
import json
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from typing import List, Optional
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from WaamClients.Laser import Laser, IOCommandBuilder, WenglorCommandBuilder

_APP_INSTANCE = None
_VIEWER_INSTANCE = None
global SUBSTRATE_MODEL
SUBSTRATE_MODEL = None

def load_params():
    default_params = {
        "TRIM_START": 10.0, "TRIM_END": 10.0, "X_MIN": 2.0, "X_MAX": 8.0,
        "PERC_MIN": 5, "DELTA_H": 1.5, "X_TOL": 1.2, "MIN_PTS": 20,
        "MIN_WIDTH": 0.2, "MIN_CURV": 0.2, "APEX_TOL": 0.5,
        "SOR_K": 7, "SOR_MULT": 0.3, "PRINT_SPEED": 10.0
    }
    try:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rt_profiles.json")
        with open(cfg_path, "r") as f:
            data = json.load(f)
            return data.get("filter", {}).get("Default", default_params)
    except:
        return default_params

def get_dout(bit: int) -> bool:
    try:
        cmd = IOCommandBuilder().read_bit(bit).build()
        response_dict = Laser.read_io(cmd)
        if response_dict and 'response' in response_dict:
            raw_resp = str(response_dict['response']).strip()
            if raw_resp == '1': return True
            if raw_resp == '0': return False
            parts = raw_resp.split(',')
            if len(parts) >= 3:
                try:
                    return int(parts[2]) == 1
                except ValueError:
                    pass
        return False
    except Exception:
        return False

def set_dout(bit: int, state: bool):
    try:
        cmd = IOCommandBuilder().write_bit(bit, state).build()
        Laser.write_io(cmd)
    except Exception:
        pass

class LiveScannerViewer(QtWidgets.QWidget):
    def __init__(self, csv_path: str, trigger_bit: int):
        super().__init__()
        self.csv_path = csv_path
        self.trigger_bit = trigger_bit
        self.layer_index = 0
        self.current_z_shift = 0.0
        self.scan_profiles = [] 
        self.raw_csv_buffer = []
        self.final_height_result = 0.0
        self.is_recording_active = False
        self.layer_finished = False
        self.trigger_seen = False
        self.scan_packet_counter = 0
        self.scan_start_time = 0.0
        self.last_gui_time = 0
        self.last_scan_timestamp = 0.0
        self.params = load_params()

        self.setWindowTitle("WaamLab: CurveFit RT Analyzer")
        self.resize(1000, 800)
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        
        self.plot_profile = pg.PlotWidget(title="Live Laser Profile")
        self.plot_profile.setLabel('left', 'Distance Z', units='mm')
        self.plot_profile.setLabel('bottom', 'Position X', units='mm')
        self.plot_profile.showGrid(x=True, y=True)
        self.plot_profile.setYRange(74, 158)
        self.curve_profile = self.plot_profile.plot(pen=pg.mkPen('c', width=2))
        
        self.plot_trend = pg.PlotWidget(title="Apex Height Trend (Real-time approx)")
        self.plot_trend.setLabel('bottom', 'Scan Packet', units='#')
        self.plot_trend.setLabel('left', 'Calculated Height', units='mm')
        self.plot_trend.showGrid(x=True, y=True)
        self.curve_trend = self.plot_trend.plot(pen=pg.mkPen('m', width=2), symbol='o', symbolSize=5)
        self.trend_data = []

        self.layout.addWidget(self.plot_profile)
        self.layout.addWidget(self.plot_trend)
        self.info_label = QtWidgets.QLabel("Status: Ready")
        self.info_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #aaa;")
        self.layout.addWidget(self.info_label)

        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
            with open(self.csv_path, "w") as f:
                f.write("Layer,ScanID,Timestamp,SensorTimestamp,PointNr,X,Z\n")
        except Exception as e:
            self.info_label.setText(f"CSV Init Error: {e}")

        try:
            Laser.write_scanner(WenglorCommandBuilder().build_setup())
        except Exception:
            pass
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(10)

    def prepare_for_new_layer(self, layer_idx: int, current_z_shift: float):
        self.layer_index = layer_idx
        self.current_z_shift = current_z_shift
        self.scan_profiles = []
        self.raw_csv_buffer = []
        self.trend_data = []
        self.curve_trend.setData([], [])
        self.final_height_result = 0.0
        self.is_recording_active = False
        self.layer_finished = False
        self.trigger_seen = False
        self.scan_start_time = 0.0
        self.scan_packet_counter = 0
        self.params = load_params()
        role = "SUBSTRATE (Layer 0)" if layer_idx == 0 else f"WELD (Layer {layer_idx})"
        self.info_label.setText(f"Waiting for trigger... [{role}]")

    def process_layer_data(self):
        global SUBSTRATE_MODEL
        if not self.scan_profiles:
            return 0.0

        p = self.params
        t_start = self.scan_profiles[0][2]
        
        valid_scans = []
        y_max_total = (self.scan_profiles[-1][2] - t_start) * p['PRINT_SPEED']
        
        for px, pz, ts in self.scan_profiles:
            y_calc = (ts - t_start) * p['PRINT_SPEED']
            if y_calc < p['TRIM_START'] or y_calc > (y_max_total - p['TRIM_END']):
                continue
            valid_scans.append((px, pz, y_calc))

        if not valid_scans:
            return 0.0

        if self.layer_index == 0:
            all_x, all_y, all_z = [], [], []
            for px, pz, py in valid_scans:
                mask_x = (px >= p['X_MIN']) & (px <= p['X_MAX'])
                if np.any(mask_x):
                    all_x.extend(px[mask_x])
                    all_y.extend(np.full(np.sum(mask_x), py))
                    all_z.extend(pz[mask_x])
            
            if not all_x: return 0.0
            
            try:
                SUBSTRATE_MODEL = NearestNDInterpolator(list(zip(all_x, all_y)), all_z)
                return float(np.mean(all_z))
            except:
                return 0.0

        else:
            if SUBSTRATE_MODEL is None:
                return 0.0
                
            robot_lift = self.current_z_shift
            all_x, all_y, all_h, all_sg = [], [], [], []

            for sg_idx, (px, pz, py) in enumerate(valid_scans):
                mask_roi = (px >= p['X_MIN']) & (px <= p['X_MAX'])
                if not np.any(mask_roi): continue
                rx = px[mask_roi]
                rz = pz[mask_roi]

                z_thresh = np.percentile(rz, p['PERC_MIN']) + p['DELTA_H']
                mask_z = rz <= z_thresh
                
                rx_f = rx[mask_z]
                rz_f = rz[mask_z]
                
                if len(rx_f) == 0: continue

                try:
                    ref_z = SUBSTRATE_MODEL(rx_f, np.full(len(rx_f), py))
                    h_local = robot_lift + (ref_z - rz_f)
                except:
                    continue

                all_x.extend(rx_f)
                all_y.extend(np.full(len(rx_f), py))
                all_h.extend(h_local)
                all_sg.extend(np.full(len(rx_f), sg_idx))

            if not all_x: return 0.0

            arr_x = np.array(all_x)
            arr_y = np.array(all_y)
            arr_h = np.array(all_h)
            arr_sg = np.array(all_sg)

            if len(arr_x) == 0: return 0.0

            coords = np.column_stack([arr_x, arr_y, arr_h]).astype(np.float32)
            tree = cKDTree(coords)
            k_val = int(p['SOR_K']) + 1
            distances, _ = tree.query(coords, k=k_val)
            mean_dist = np.mean(distances[:, 1:], axis=1)
            
            global_mean = np.mean(mean_dist)
            global_std = np.std(mean_dist)
            threshold = global_mean + p['SOR_MULT'] * global_std
            mask_sor = mean_dist <= threshold

            valid_x = arr_x[mask_sor]
            valid_y = arr_y[mask_sor]
            valid_h = arr_h[mask_sor]
            valid_sg = arr_sg[mask_sor]

            if len(valid_x) == 0: return 0.0

            med_x = np.median(valid_x)
            mask_xtol = (valid_x >= (med_x - p['X_TOL'])) & (valid_x <= (med_x + p['X_TOL']))

            valid_x = valid_x[mask_xtol]
            valid_h = valid_h[mask_xtol]
            valid_sg = valid_sg[mask_xtol]

            if len(valid_x) == 0: return 0.0

            temp_apexes = []
            for sg in np.unique(valid_sg):
                mask = valid_sg == sg
                xg = valid_x[mask]
                hg = valid_h[mask]

                if len(xg) < p['MIN_PTS']: continue
                if xg.max() - xg.min() < p['MIN_WIDTH']: continue

                try:
                    h_max_local = hg.max()
                    cap_mask = hg >= (h_max_local - 1.0)
                    if np.sum(cap_mask) >= 3:
                        xg_fit, hg_fit = xg[cap_mask], hg[cap_mask]
                    else:
                        xg_fit, hg_fit = xg, hg
                        
                    coeffs = np.polyfit(xg_fit, hg_fit, 2)
                    a, b, c = coeffs
                    
                    if abs(a) > 1e-6:
                        vx = -b / (2*a)
                    else:
                        vx = xg.mean()

                    if not (p['X_MIN'] <= vx <= p['X_MAX']): continue

                    vz = a*(vx**2) + b*vx + c
                    if abs(vz - h_max_local) > 0.3:
                        vz = h_max_local

                    if a >= -p['MIN_CURV']: continue

                    temp_apexes.append({'x': vx, 'z': vz})
                except:
                    continue

            if not temp_apexes: return 0.0

            apex_x = np.array([ap['x'] for ap in temp_apexes])
            apex_z = np.array([ap['z'] for ap in temp_apexes])

            med_apex_x = np.median(apex_x)
            mask_apex_tol = (apex_x >= med_apex_x - p['APEX_TOL']) & (apex_x <= med_apex_x + p['APEX_TOL'])
            final_apex_z = apex_z[mask_apex_tol]

            if len(final_apex_z) == 0: return 0.0

            return float(np.median(final_apex_z))
        
    def save_buffer_to_csv(self):
        if not self.raw_csv_buffer:
            return
        try:
            with open(self.csv_path, "a") as f:
                f.writelines(self.raw_csv_buffer)
        except:
            pass
            
    def update_gui(self):
        try:
            now = time.perf_counter()
            dout_active = get_dout(self.trigger_bit)
            
            if dout_active:
                if not self.trigger_seen:
                    self.scan_start_time = now
                    if not self.is_recording_active:
                        self.trend_data = []
                        self.curve_trend.setData([], [])
                self.trigger_seen = True
                self.is_recording_active = True
            
            if self.trigger_seen and not dout_active and not self.layer_finished:
                self.is_recording_active = False
                self.layer_finished = True
                self.final_height_result = self.process_layer_data()
                self.info_label.setText(f"L{self.layer_index} Done. H: {self.final_height_result:.2f} mm")
                self.save_buffer_to_csv()

            scan_data = Laser.read_scanner()
            
            if scan_data is not None and len(scan_data['x']) > 0:
                current_ts = scan_data.get('timestamp', 0)
                if current_ts == self.last_scan_timestamp and current_ts != 0:
                    return
                
                self.last_scan_timestamp = current_ts
                x = np.array(scan_data['x'])
                z = np.array(scan_data['z'])
                
                if self.is_recording_active:
                    self.scan_packet_counter += 1
                    self.scan_profiles.append((x, z, now))
                    
                    lines = []
                    step = 1                
                    for i in range(0, len(x), step):
                        lines.append(f"{self.layer_index},{self.scan_packet_counter},{now:.3f},{current_ts},{i},{x[i]:.4f},{z[i]:.4f}\n")
                    self.raw_csv_buffer.extend(lines)

                    if (now - self.last_gui_time) > 0.1:
                        mask_center = (x > 4.0) & (x < 6.0)
                        if np.any(mask_center):
                            est_h = 158 - np.min(z[mask_center])
                            self.trend_data.append(est_h)
                            self.curve_trend.setData(self.trend_data)
                        
                        self.curve_profile.setData(x, z)
                        status = "REC" if self.is_recording_active else "IDLE"
                        self.info_label.setText(f"Packets: {len(self.scan_profiles)} | {status}")
                        self.last_gui_time = now

        except Exception:
            pass

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

def pre_initialize_viewer(csv_path: str, trigger_bit: int):
    global _APP_INSTANCE, _VIEWER_INSTANCE
    if _APP_INSTANCE is None:
        _APP_INSTANCE = QtWidgets.QApplication.instance()
        if _APP_INSTANCE is None:
            _APP_INSTANCE = QtWidgets.QApplication(sys.argv)
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'd')

    if _VIEWER_INSTANCE is None:
        _VIEWER_INSTANCE = LiveScannerViewer(csv_path=csv_path, trigger_bit=trigger_bit)

def run_scan_cycle(cur_layer: int, job_name: str, base_z_ref: Optional[float] = None, current_z_shift: float = 0.0, trigger_bit: int = 34) -> float:
    global _APP_INSTANCE, _VIEWER_INSTANCE
    if _APP_INSTANCE is None:
        _APP_INSTANCE = QtWidgets.QApplication.instance()
        if _APP_INSTANCE is None:
            _APP_INSTANCE = QtWidgets.QApplication(sys.argv)

    if _VIEWER_INSTANCE is None:
        _VIEWER_INSTANCE = LiveScannerViewer(csv_path=job_name, trigger_bit=trigger_bit)
        _VIEWER_INSTANCE.show()
    else:
        if not _VIEWER_INSTANCE.isVisible():
             _VIEWER_INSTANCE.show()

    _VIEWER_INSTANCE.prepare_for_new_layer(cur_layer, current_z_shift)
    
    while not _VIEWER_INSTANCE.layer_finished:
        _APP_INSTANCE.processEvents()
        time.sleep(0.005)
        if not _VIEWER_INSTANCE.isVisible():
            return 0.0

    return _VIEWER_INSTANCE.final_height_result

if __name__ == "__main__":                  
    pre_initialize_viewer("Manual_Test_points.csv", 34)
    for l in range(0, 3):
        h = run_scan_cycle(l, "Manual_Test_points.csv", None, l*0.9, 34)
        time.sleep(2)