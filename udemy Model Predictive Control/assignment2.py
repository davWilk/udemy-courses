from distutils.log import error
import numpy as np
from sim.sim2d import sim_run
from math import sin, cos, tan, pi, atan2
# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['OBSTACLES'] = False

class ModelPredictiveControl:
    def __init__(self):
        # Solver settings
        self.horizon = 20
        self.dt = 0.2

        # Setting limits            
        self.speed_lim_kph = 10 #  Assuming we are in a car park!

        self.pedal_upper_lim = 5
        self.pedal_lower_lim = -5

        self.steering_upper_lim = 0.8
        self.steering_lower_lim = -0.8
 
        # Reference or set point the controller will achieve.
        self.reference1 = [10, 10, 0]
        self.reference2 = [10, 2, 3*3.14/2 - 0.6]

    def plant_model(self,prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3]
        a_t = pedal # assume 1:1 ratio between pedal and acceleration

        # Iterative calculation of state
        x_t += (v_t * np.cos(psi_t)) * dt  # x = x + x_dot
        y_t += (v_t * np.sin(psi_t)) * dt  # y = y + y_dot
        psi_t += (np.tan(steering)/2.5) * dt * v_t # psi = psi + psi_dot

        # if psi_t_1 % (pi) > pi:
        #     psi_t_1 = ((psi_t_1) % (2*pi)) - 2*pi
        # else:
        #     psi_t_1 = ((psi_t_1) % (2*pi)) 

        # Velocity model accounts for resistance to motion, constant of -1/25*v_t
        v_t_1 = 0.96 * v_t + a_t * dt
        return [x_t, y_t, psi_t, v_t_1]

    def cost_function(self,u, *args):
        state = args[0]
        ref = args[1]
        cost = 0.0
        ref[2] = self.standardize_psi_ref(ref[2])

        for k in range(self.horizon):
            v_t = u[2*k]
            psi_t =  u[2*k+1]

            state = self.plant_model(state, self.dt, u[2*k], u[2*k+1])

            error_phi_abs =  abs(ref[2] - state[2])#** 2
            error_x_abs = abs(ref[0] - state[0])
            error_y_abs = abs(ref[1] - state[1])

            cost += 0.5 * error_x_abs**2 + 10* error_x_abs
            cost += 0.5 * error_y_abs**2 + 10*error_y_abs
            
            if error_phi_abs < 0.3:
                cost += 1.75* error_phi_abs 
            else:
                cost += 2.5*error_phi_abs**2 + error_phi_abs

            # Speed cost - Large penalty for speeds > 10kph
            speed_kph = state[3] * 3.6
            if abs(speed_kph) > self.speed_lim_kph:
                cost += abs(speed_kph  - self.speed_lim_kph) * 10
            
            # Steering position cost - VERY large penalty for exceeding bounds of wheel
            #  Steering wheel movement is constrained to +/- 0.8rad
            if u[2*k+1] > self.steering_upper_lim:
                cost += abs(u[2*k+1] - self.steering_upper_lim) * 10
            elif u[2*k+1] < self.steering_lower_lim:
                cost += abs(u[2*k+1] - self.steering_lower_lim) * 10



        # Must implement some randomization to remove local minimum!!
        # IF state has not changed across complete horizon, could be in a local minimum
        #  thus manually inject random plant input to try overcome this
        # if self.is_local_minimum(u, state, ref)
        #     self.randomize_input(u)
        #     cost = self.cost_function(...)
        
        return cost

sim_run(options, ModelPredictiveControl)
