import numpy as np
from sim.sim1d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['FULL_RECALCULATE'] = False

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 20
        self.dt = 0.2

        # Reference or set point the controller will achieve.
        self.reference = [50, 0, 0]

    def plant_model(self, prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        v_t = prev_state[3] # m/s

        a_t = pedal

        x_t_1 = x_t + v_t*dt
        v_t_1 = 0.96*v_t + a_t*dt
        return [x_t_1, 0, 0, v_t_1]

    def cost_function(self,u, *args):
        state = args[0]
        ref = args[1]
        cost = 0.0
        for k in range(self.horizon):
            state = self.plant_model(state, self.dt, u[2*k], u[2*k+1])
            cost += (state[0] - ref[0])**2 

            speed_kph = state[3] * 3.6
            if abs(speed_kph) > 10:
                cost += 100 * speed_kph

            if ref[0] - state[0] < 0:
                cost += (state[0] - ref[0])*1000 
        return cost

sim_run(options, ModelPredictiveControl)
