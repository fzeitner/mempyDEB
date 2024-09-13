
from .derivatives import *
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

def simulate_DEBBase(params):
    # maximum srutural length (mass ^ (1/3))
    # used to calculate size-dependent elimination rate
    
    glb = params[0].copy()
    spc = params[1].copy()

    simout = pd.DataFrame()
    LS_max = (spc['kappa']*spc['Idot_max_rel']*spc['eta_IA'])/spc['k_M']

    y0 = [1e-5, 0, spc["X_emb_int"], glb['Xdot_in'], 0., 0.]
    t_eval = np.arange(
            glb['tspan'][0], 
            glb['tspan'][1]+1,
            )
        
    sol = solve_ivp(
        DEBBase, glb['tspan'], y0, 
        t_eval = t_eval,
        rtol = 1e-6, 
        args = (glb, spc, LS_max)
        ) 
    
    # calculation of survival over time based on GUTS-SD
    stvec = [np.exp(-LL2h(D, (spc['ED50_h'], spc['beta_h'])) * t) for (t,D) in zip(sol.t, sol.y[5])]
        
    sim = pd.DataFrame(np.transpose(sol.y))
    sim.rename({0 : 'S', 1 : 'R', 2 : 'X_emb', 3 : 'X', 4 : 'D_j', 5 : 'D_h'}, axis = 1, inplace = True)
    sim['t'] = np.array(sol.t) - glb['a_int'] # subtracting initial age from time helps to align output with experimental observaions
    sim['survival'] = stvec

    return sim