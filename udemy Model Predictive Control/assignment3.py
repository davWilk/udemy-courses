from pyexpat.errors import XML_ERROR_CANT_CHANGE_FEATURE_ONCE_PARSING
from re import X
import numpy as np
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['OBSTACLES'] = True

class Point:
    def  __init__(self, r, theta):
        self.r = r
        self.theta = theta

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 2
        self.dt = 0.2

        self.pedal_upper_lim = 5
        self.pedal_lower_lim = -5

        self.steering_upper_lim = 0.8
        self.steering_lower_lim = -0.8

        # Reference or set point the controller will achieve.
        self.reference1 = [10, 0, 0]
        self.reference2 = [2, 4, 3.14/3]

        self.x_obs = 5
        self.y_obs = 0.1


        # Array of points around vehicle where distance
        # sensors would be placed.
        # Required for enhanced obstacle avoidance.
        # These points are all RELATIVE to the mid-rear of vehicle, and are
        #  modelled as if the vehicle was facing to the right. 
        # See below diagram:
        #
        #   #-----1------#
        #   |            |
        #   O  (2.5x1)   2
        #   |            |
        #   #-----3------#
        # The points are in (r,theta) form thus angles need to be added to current vehicle angle
        #  to get ABSOLUTE points

        # 7 sensors behaves on par with having 4 sensors
        # self.sensor_points = [Point(0,0), Point(0.5, 1.571), Point(1.346, 0.381), Point(2.550, 0.197), 
        #                          Point(2.5, 0), Point(2.550, -0.197), Point(1.346, -0.381), Point(0.5, -1.571)]

        self.sensor_points = [Point(0,0), Point(1.346, 0.381), 
                                Point(2.5, 0), Point(1.346, -0.381)]

    def plant_model(self,prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3]

        # Assumption of 1:1 relationship between pedal/acceleration and steering/wheel angle
        a_t = pedal
        beta = steering

        v_t = 0.96*v_t + a_t * dt
        x_t += (v_t * np.cos(psi_t)) * dt
        y_t += (v_t * np.sin(psi_t)) * dt

        psi_t += (v_t * np.tan(beta) / 2.5) * dt

        return [x_t, y_t, psi_t, v_t]

    def cost_function(self,u, *args):
        state = args[0]
        ref = args[1]
        cost = 0.0

        for k in range(self.horizon):
            v_t_init = state[2]

            state = self.plant_model(state, self.dt, u[2*k], u[2*k+1])

            phi_t = state[2]
            x_t = state[0]
            y_t = state[1]
            v_t = state[3]

            error_phi_abs =  abs(ref[2] - state[2])
            error_x_abs = abs(ref[0] - state[0])
            error_y_abs = abs(ref[1] - state[1])
            # x/y costs
            cost += 0.5*error_x_abs ** 2 + 5*error_x_abs
            cost += 0.5*error_y_abs ** 2 + 5*error_y_abs

            # angle cost
            cost += 1.5*error_phi_abs**2# + error_phi_abs

            # Steering position cost - VERY large penalty for exceeding bounds of wheel
            #  Steering wheel movement is constrained to +/- 0.8rad
            if u[2*k+1] > self.steering_upper_lim:
                cost += abs(u[2*k+1] - self.steering_upper_lim) * 10
            elif u[2*k+1] < self.steering_lower_lim:
                cost += abs(u[2*k+1] - self.steering_lower_lim) * 10

            # Pedal position cost
            if u[2*k] > self.pedal_upper_lim:
                cost += abs(u[2*k] - self.pedal_upper_lim) * 10
            if u[2*k] < self.pedal_lower_lim:
                cost += abs(u[2*k] - self.pedal_lower_lim) * 10


            # penalty for approaching obstacle            
            # Cost is implemented such that cost increases much
            # faster as vehicle approaches the obstacle.
            # This should deter the optimizer from
            # solving to such position.
            dist = 0
            for point in self.sensor_points:
                _x = x_t + (point.r * np.cos(point.theta + phi_t))
                _y = y_t + (point.r * np.sin(point.theta + phi_t))

                x_dist = abs(self.x_obs - _x)
                y_dist = abs(self.y_obs - _y)
                dist += np.sqrt((x_dist**2)+(y_dist**2))

                # Protecting for div by zero
                if dist < 0.00001:
                    dist = 0.00001
                cost += 20/(dist)
            
            # Speed limit penalty
            speed_kph = v_t * 3.6
            if abs(speed_kph) > 15:
                cost += (abs(speed_kph)  - 15) * 1

            # Acceleration cost
            cost += (state[3] - v_t_init) ** 2 * 0.1
            

        return cost

sim_run(options, ModelPredictiveControl)
