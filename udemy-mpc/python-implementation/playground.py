import numpy as np
from sim.sim_play import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['OBSTACLES'] = False

class Run:
    def __init__(self):
        # Solver time step
        self.dt = 0.1

        # Reference or set point the controller will achieve.
        #  NEEDED for wustangdan's sim to run
        self.reference1 = [10, 0, 0]
        self.reference2 = None #[10, 2, 3.14/2]

        # Error summation for Integral action
        self.error_sum_x = 0
        self.error_sum_y = 0

        # Storing previous position to allow simple Derivative action
        self.prev_x_err = 0
        self.prev_y_err = 0
        self.prev_phi = 0
        
        self.pedal_upper_lim = 5
        self.pedal_lower_lim = -5

        self.steering_upper_lim = 0.8
        self.steering_lower_lim = -0.8

        # PID Gains
        self.Kp = 0.3
        self.Ki = 0.0
        self.Kd = 0.85  

    def run(self, current_state):
        x_t = current_state[0] # X Location [m]
        y_t = current_state[1] # Y Location [m]
        psi_t = current_state[2] # Angle [rad]
        v_t = current_state[3] # Speed [m/s]
        
        steering = 0 # Max; 0.8, Min: -0.8

        pedal = self.pid_ctrl(current_state)


        if pedal > self.pedal_upper_lim:
            pedal = self.pedal_upper_lim
        elif pedal < self.pedal_lower_lim:
            pedal = self.pedal_lower_lim

        return [pedal, steering]
    
    def pid_ctrl(self, current_state):
        # Only implemented in one dimension for now
        x_diff = self.reference1[0] - current_state[0]
        y_diff = self.reference1[1] - current_state[1]
        
        # Update integral data
        self.error_sum_x += x_diff * self.dt
        self.error_sum_y += y_diff * self.dt

        # Update derivative data
        error_Kd_x = (x_diff - self.prev_x_err) / self.dt
        error_Kd_y = (y_diff - self.prev_y_err) / self.dt

        self.prev_x_err = x_diff
        self.prev_y_err = y_diff

        # Return controller output
        return (x_diff * self.Kp) + (self.error_sum_x * self.Ki) + (error_Kd_x * self.Kd)




sim_run(options, Run)
