
from .derivatives import *
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import copy

def simulate_DEBBase(params):
    # maximum srutural length (expressed in mass ^ (1/3))
    # this is used in the derivatives to compute size-dependent TK
    LS_max = (params.spc['kappa']*params.spc['Idot_max_rel']*params.spc['eta_IA_0'])/params.spc['k_M_0']

    y0 = [1e-5, 0, params.spc["X_emb_int"], params.glb['Xdot_in'], 0., 0.] # defining initial states              ### second 0. is initial CD

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
        args = (params.glb, params.spc, LS_max, ) # additional arguments to make available within the model function
        ) 
    
    sim = pd.DataFrame(np.transpose(sol.y))
    sim.rename({
        0 : 'S', 
        1 : 'R', 
        2 : 'X_emb', 
        3 : 'X', 
        4 : 'D_j',
        5 : 'Cd_in'
        }, axis = 1, inplace = True)
    
    # subtracting initial age from time helps to align output with experimental observaions
    sim['t'] = np.array(sol.t) - params.glb['a_int'] 

    return sim

def constant_exposures(simulator, params, C_Wvec):
    p = copy.deepcopy(params)
    output = pd.DataFrame() # colelct all output in this data frame

    for C_W in C_Wvec:
        p.glb["C_W"] = C_W
        out_i = simulator(p).assign(C_W = C_W)
        output = pd.concat([output, out_i])

    return output


def EulerExplnD(func,dim,pars,h,func_init,tend):
    tvals = np.linspace(0,tend,int(np.ceil(tend/h)+1))
    funcvals = np.zeros((len(tvals), dim))
    old_vals = func_init
    funcvals[0] = old_vals
    for step in np.arange(1,len(tvals)):
        t_now = tvals[step]
        for dimix in np.arange(0,dim):
            funcvals[step,dimix] = old_vals[dimix]+h*(func(t_now,old_vals,*pars)[dimix])
            old_vals = funcvals[step]
    return (tvals,funcvals)

def simulate_DEBBase_Cd(params):
    # maximum srutural length (expressed in mass ^ (1/3))
    # this is used in the derivatives to compute size-dependent TK
    LS_max = (params.spc['kappa']*params.spc['Idot_max_rel']*params.spc['eta_IA_0'])/params.spc['k_M_0']

    y0 = [1e-5, 0, params.spc["X_emb_int"], params.glb['Xdot_in'], 0.] # defining initial states              ### second 0. is initial CD

    # defining time-points to evaluate
    # this comes at some computational cost compared to letting the ODE solver decide, 
    # but we need it if we want to compare output to experimental data
    t_eval = np.arange( 
            params.glb['tspan'][0], 
            params.glb['tspan'][1]+1,
            )
        
    # this defines the initial value problem to solve
    
    sol = solve_ivp(
        DEBBase_Cd, # the model  to solve
        params.glb['tspan'], # time span to simulate
        y0, # initial states
        t_eval = t_eval, # time points to evaluate
        
        rtol = 1e-6, # relative tolerance
        args = (params.glb, params.spc, LS_max, ), # additional arguments to make available within the model function
        min_step = 0.1
        ) 
    """
    sol = np.array(EulerExplnD(DEBBase_Cd, 5, pars=(params.glb, params.spc, LS_max, ), h=0.01, func_init=y0, tend=params.glb['tspan'][1]+1))
    """
    sim = pd.DataFrame(np.transpose(sol.y))
    sim.rename({
        0 : 'S', 
        1 : 'R', 
        2 : 'X_emb', 
        3 : 'X', 
        4 : 'Cd_in'
        }, axis = 1, inplace = True)
    
    # subtracting initial age from time helps to align output with experimental observaions
    sim['t'] = np.array(sol.t) - params.glb['a_int'] 

    return sim