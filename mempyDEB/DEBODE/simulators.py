
from .derivatives import *
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

def simulate_DEBBase(params):
    # maximum srutural length (expressed in mass ^ (1/3))
    # this is used in the derivatives to compute size-dependent TK
    LS_max = (params.spc['kappa']*params.spc['Idot_max_rel']*params.spc['eta_IA_0'])/params.spc['k_M_0']

    y0 = [1e-5, 0, params.spc["X_emb_int"], params.glb['Xdot_in'], 0., 0.] # defining initial states

    # defining time-points to evaluate
    # this comes at some computational cost compared to letting the ODE solver decide, 
    # but we need it if we want to compare output to experimental data
    t_eval = np.arange( 
            params.glb['tspan'][0], 
            params.glb['tspan'][1]+1,
            )
        
    # this defines the initial value problem to solve
    sol = solve_ivp(
        DEBBase, # the model  to solve
        params.glb['tspan'], # time span to simulate
        y0, # initial states
        t_eval = t_eval, # time points to evaluate
        rtol = 1e-6, # relative tolerance
        args = (params.glb, params.spc, LS_max) # additional arguments to make available within the model function
        ) 
    
    # calculation of survival over time based on GUTS-SD
    stvec = [np.exp(-LL2h(D, (params.spc['ED50_h'], params.spc['beta_h'])) * t) for (t,D) in zip(sol.t, sol.y[5])]
        
    yhat = pd.DataFrame(np.transpose(sol.y))
    yhat.rename({0 : 'S', 1 : 'R', 2 : 'X_emb', 3 : 'X', 4 : 'D_j', 5 : 'D_h'}, axis = 1, inplace = True)
    yhat['t'] = np.array(sol.t) - params.glb['a_int'] # subtracting initial age from time helps to align output with experimental observaions
    yhat['survival'] = stvec

    return yhat