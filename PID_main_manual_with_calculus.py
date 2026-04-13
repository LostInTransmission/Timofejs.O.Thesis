import sys
import os
import time
import csv
import datetime
from WaamClients.Laser import Laser, IOCommandBuilder

import RT_PL_logic_main as Scan_data_Percentile_RT
from PID.PID_controller import HeightPIDController

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.getcwd())

MANUAL_TARGET_HEIGHT = 0.9
MANUAL_PRINT_SPEED = 10
JOB_NAME_LOG = "Manual_Test_points_ALL_POINTS.csv"

PID_KP = 3.0 
PID_KI = 0.5
PID_KD = 0.0 

PID_START_LAYER = 3

BIT_LAYER_SYNC = 35
BIT_SCAN_TRIGGER = 34
BIT_SPEED_APPLY = 36

SPEED_MIN_PCT = 60
SPEED_MAX_PCT = 140

LOG_DIR = "PID_Logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def create_log_file():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(LOG_DIR, f"InterLayer_Log_{timestamp}.csv")
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "Layer_Index", "Role",
            "Val_Measured", "Base_Ref_Z", 
            "H_Actual_Total", "H_Target_Total", 
            "Error_Total", "Calculated_Speed_Pct", "PID_I_Term"
        ])
    return filename

def log_data(filename, data_row):
    try:
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(data_row)
    except Exception as e:
        print(f"[LOG ERROR] Could not write to log: {e}")

def set_robot_speed_override(speed_percent: int):
    if speed_percent < SPEED_MIN_PCT: speed_percent = SPEED_MIN_PCT
    if speed_percent > SPEED_MAX_PCT: speed_percent = SPEED_MAX_PCT
    
    cmd = IOCommandBuilder().speed(int(speed_percent)).build()
    Laser.write_io(cmd)

def main():
    print("=== INTER-LAYER CONTROLLER STARTED ===")
    print(f"Target Layer H: {MANUAL_TARGET_HEIGHT}mm")
    print(f"PID: Kp={PID_KP}, Ki={PID_KI}, Limits={SPEED_MIN_PCT}-{SPEED_MAX_PCT}%")
    
    set_robot_speed_override(100)
    
    pid = HeightPIDController(kp=PID_KP, ki=PID_KI, kd=PID_KD, min_scale=SPEED_MIN_PCT, max_scale=SPEED_MAX_PCT)
    
    current_layer_index = -1
    base_ref_z = None
    calculated_next_speed = 100 
    
    speed_applied = False 
    
    log_file = create_log_file()

    try:
        while True:
            sig_new_layer = Scan_data_Percentile_RT.get_dout(BIT_LAYER_SYNC)
            sig_scanning = Scan_data_Percentile_RT.get_dout(BIT_SCAN_TRIGGER)
            sig_welding = Scan_data_Percentile_RT.get_dout(BIT_SPEED_APPLY)
            
            if sig_new_layer:
                Scan_data_Percentile_RT.set_dout(BIT_LAYER_SYNC, False)
                current_layer_index += 1
                print(f"\n>>> LAYER {current_layer_index} DETECTED")
                
                speed_applied = False 
                
                while Scan_data_Percentile_RT.get_dout(BIT_LAYER_SYNC):
                    time.sleep(0.1)

            if sig_scanning:
                if base_ref_z is None:
                    current_layer_index = 0
                    print(f"[INIT] First scan detected. Force setting Layer Index to 0 (Substrate).")
                
                print(f"[SCAN] Detected Trigger.")
                
                robot_z_lift = 0.0
                if current_layer_index > 1:
                    robot_z_lift = (current_layer_index - 1) * MANUAL_TARGET_HEIGHT
                
                scan_result = Scan_data_Percentile_RT.run_scan_cycle(
                    cur_layer=current_layer_index, 
                    job_name=JOB_NAME_LOG, 
                    base_z_ref=base_ref_z,
                    current_z_shift=robot_z_lift,
                    trigger_bit=BIT_SCAN_TRIGGER
                )
                
                is_valid = True
                
                if base_ref_z is None:
                    if scan_result < 50.0: 
                        is_valid = False
                else:
                    if scan_result <= 0.0001: 
                        is_valid = False

                if not is_valid:
                    print(f"[ERR] ALARM! Invalid Scan Result: {scan_result:.4f}. Ignoring layer data.")
                    log_data(log_file, [
                        datetime.datetime.now().strftime("%H:%M:%S"),
                        current_layer_index, "INVALID_READ",
                        f"{scan_result:.2f}", "ERR", "ERR", "ERR", "ERR", 
                        int(calculated_next_speed), "ERR"
                    ])
                    time.sleep(1.0) 
                    continue
                
                role = "UNKNOWN"
                h_total_actual = 0.0
                h_total_target = 0.0
                error = 0.0
                
                if base_ref_z is None:
                    base_ref_z = scan_result
                    print(f"[CALIB] Base Reference Set: {base_ref_z:.4f}")
                    calculated_next_speed = 100
                    role = "SUBSTRATE_ZERO"
                    h_total_actual = 0.0
                else:
                    role = "MEASUREMENT"
                    h_total_actual = scan_result
                    
                    layers_welded_count = current_layer_index
                    h_total_target = layers_welded_count * MANUAL_TARGET_HEIGHT
                    
                    if h_total_target < 0: h_total_target = 0
                    if h_total_actual < 0: h_total_actual = 0
                    
                    error = h_total_actual - h_total_target
                    
                    print(f"[PID] Act: {h_total_actual:.2f} | Tgt: {h_total_target:.2f} | Err: {error:.2f}")
                    
                    if layers_welded_count >= PID_START_LAYER:
                        calculated_next_speed = pid.compute_speed_override(h_total_actual, h_total_target)
                        print(f"[PID] Speed for Next Layer: {int(calculated_next_speed)}%")
                    else:
                        print(f"[INFO] Warmup Layer {layers_welded_count}. Speed held 100%.")
                        calculated_next_speed = 100
                
                log_data(log_file, [
                    datetime.datetime.now().strftime("%H:%M:%S"),
                    current_layer_index, role,
                    f"{scan_result:.2f}", f"{base_ref_z if base_ref_z else 0:.2f}",
                    f"{h_total_actual:.2f}", f"{h_total_target:.2f}",
                    f"{error:.2f}", int(calculated_next_speed),
                    f"{pid.integral:.2f}"
                ])

            if sig_welding:
                if not speed_applied:
                    print(f"[WELD] Applying Speed: {int(calculated_next_speed)}%")
                    #set_robot_speed_override(int(calculated_next_speed))             #Comment to stop PID from sending new speed to the roboto
                    speed_applied = True
            else:
                if speed_applied:
                    set_robot_speed_override(100)
                    speed_applied = False
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n Stopped by user.")
        set_robot_speed_override(100)
    except Exception as e:
        print(f"Critical error: {e}")

if __name__ == "__main__":
    main()