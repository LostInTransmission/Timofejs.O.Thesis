class HeightPIDController:
    def __init__(self, kp, ki, kd, min_scale=50.0, max_scale=150.0): # Limited to 50-150% by default
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Speed Override limits (robots usually accept 0-255, but 50-150% is reasonable)
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        self.prev_error = 0.0
        self.integral = 0.0
        
        # last_scale_factor removed, memory is now entirely in self.integral

    def compute_speed_override(self, current_height, target_height, planned_speed=None):
        """
        Calculates the absolute value of Speed Override (%).
        Base value is always 100%. PID adds or subtracts % from the base.
        
        planned_speed: Kept for compatibility, but not used in calculations here,
                       as we regulate the relative % (Scale), not the absolute speed.
        """
        
        # 1. Error calculation
        # In WAAM: If height (current) > target -> over-deposited -> need to move FASTER.
        # Thus, for a positive error (current > target), speed should increase.
        # Error > 0 -> Output must increase.
        error = current_height - target_height

        # --- DEADBAND ---
        # If the error is small, treat it as noise or acceptable tolerance.
        # This prevents speed jitter and accumulation of micro-errors in the integral.
        ERROR_TRESHOLD = 0.35
        if abs(error) < ERROR_TRESHOLD:
            error = 0.0
        # -------------------------------

        # 2. Proportional term
        p_term = self.kp * error

        # 3. Integral term (Accumulates history to maintain speed adjustments if needed)
        # If error is zeroed (in deadband), integral remains unchanged (Hold state).
        self.integral += error
        
        # Anti-windup protection
        # Limit integral to reasonable bounds to prevent it from growing to infinity
        # For example, keeping integral contribution within +/- 50% of speed
        integration_limit = 50.0 / (self.ki if self.ki > 0 else 1.0)
        self.integral = max(-integration_limit, min(self.integral, integration_limit))
        
        i_term = self.ki * self.integral

        # 4. Derivative term
        d_term = self.kd * (error - self.prev_error)

        # 5. Final percentage calculation
        # Base = 100%. PID determines how much to add/subtract.
        output_correction = p_term + i_term + d_term
        
        new_scale_percent = 100.0 + output_correction

        # 6. Saturation (Output limits)
        new_scale_percent = max(self.min_scale, min(new_scale_percent, self.max_scale))

        # Save error for the next D-term calculation
        self.prev_error = error

        return new_scale_percent

    def reset(self):
        """Resets PID state (e.g., when starting a new part)"""
        self.prev_error = 0.0
        self.integral = 0.0