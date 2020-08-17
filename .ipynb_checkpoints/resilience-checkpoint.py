# resilience class

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class ResilientWind():

    def __init__(self,fi,power_agc,lam):

        # FLORIS parameters
        self.fi = fi
        self.nTurbs = len(fi.get_turbine_power())

        # optimization parameters
        self.lam1 = lam
        self.lam2 = 1 - lam

        # grid signal parameters
        self.power_agc = power_agc
        self.Nt = len(power_agc)

        # outputs
        self.power_initial = 0
        self.power_opt = 0
        self.ai = np.zeros(self.nTurbs)

    def power(self,x,power_agc):

        # compute the power of FLORIS using axial induction factors
        # TODO: translates to blade pitch and tip speed ratios - actual inputs to a wind turbine
        self.fi.reinitialize_flow_field()
        self.fi.calculate_wake(axial_induction=x)

        # lam1 - weight getting the signal correct, lam2 - weight a lower TI which corresponds to lower loads
        output =   (self.lam1 / (power_agc)) * (self.fi.get_farm_power() - power_agc)**2 \
                 + (self.lam2 / (0.1)) * np.linalg.norm(self.fi.get_turbine_ti())**2

        # print(x, power_agc, self.fi.get_farm_power(),output)

        return output

    def power_est(self,x,power_agc,loss_comms):

        # compute the power of FLORIS using axial induction factors
        # TODO: translates to blade pitch and tip speed ratios - actual inputs to a wind turbine
        self.fi.reinitialize_flow_field()
        self.fi.calculate_wake(axial_induction=x)

        # lam1 - weight getting the signal correct, lam2 - weight a lower TI which corresponds to lower loads
        turb_powers = self.fi.get_turbine_power()
        turb_powers1 = np.delete(turb_powers, loss_comms, 0)
        avg_power = np.mean(turb_powers1)

        # estimate farm power = sum of all turbines in communication + estimate (average) of the other turbines
        power_farm_est = np.sum(turb_powers1) + len(loss_comms) * avg_power
        output =   (self.lam1 / (power_agc)) * (power_farm_est - power_agc)**2 \
                 + (self.lam2 / (0.1)) * np.linalg.norm(self.fi.get_turbine_ti())**2

        # print(x, avg_power, turb_powers[0], np.sum(turb_powers1), power_agc, power_farm_est)

        return output

    def optimize(self,ws,power_agc,opt_type='default',loss_comms=[]):

        self.fi.reinitialize_flow_field(wind_speed=ws)

        # intial condition - all turbines are operating at max axial induction factors
        x0 = np.ones(self.nTurbs)

        # set bounds
        bnds = []
        ai = []
        lower_ai = 0.0000001
        eta = 0.77
        for j in range(self.nTurbs):
            Cp  = self.fi.floris.farm.flow_field.turbine_map.turbines[j].fCpInterp(ws)
            roots = np.roots([4,-8,4,-Cp/eta])
            upper_ai = roots[2]
            ai.append(upper_ai)
            if opt_type == 'default':
                bnds.append((lower_ai,upper_ai))
            elif opt_type == 'loss_comms':
                if j in loss_comms:
                    bnds.append((upper_ai, upper_ai))
                else:
                    bnds.append((lower_ai, upper_ai))
            elif opt_type == 'loss_comms_est':
                if j in loss_comms:
                    bnds.append((upper_ai, upper_ai))
                else:
                    bnds.append((lower_ai, upper_ai))
            elif opt_type == 'shutdown':
                if j in loss_comms:
                    bnds.append((lower_ai, lower_ai))
                else:
                    bnds.append((lower_ai, upper_ai))

        self.fi.calculate_wake(axial_induction=ai)
        self.power_initial = self.fi.get_farm_power()

        # optimize timestep
        if opt_type == 'loss_comms_est':
            res = minimize(self.power_est, x0, args=(power_agc,loss_comms), method='L-BFGS-B', bounds=bnds)
        else:
            res = minimize(self.power, x0, args=(power_agc), method='L-BFGS-B', bounds=bnds)

        # print(res)

        # evaluate floris with the optimized settings
        self.fi.reinitialize_flow_field(wind_speed=ws)
        self.fi.calculate_wake(axial_induction=res.x)
        self.power_opt = self.fi.get_farm_power()

        # print(power_agc,self.power_opt)




