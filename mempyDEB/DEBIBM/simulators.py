import numpy as np
import pandas as pd
from collections import namedtuple
import mesa
from tqdm import tqdm
from pathlib import Path
import os

from .defaultparams import *
from .model import *

# simulator for exposures with replicates

IBMOut = namedtuple('IBMout', 'mout aout')

def simulate_DEBIBM(p):

    glb, spc = p

    # For parameters which are subject to individual variability, 
    # the initial values are set equal to the mean
    # this is only relevant for initializing the first cohort
    # the individual variability method will be called afterwards
    spc['Idot_max_rel'] = spc['Idot_max_rel_mean']
    spc['S_p'] = spc['S_p_mean']
    spc['X_emb_int'] = spc['X_emb_int_mean']
    
    tmax_steps = int(np.ceil(glb['tspan'][1] * glb['tres'])) # calculate maximum time in terms of model steps
    
    mout_tot = pd.DataFrame() # initialize output dataframes
    aout_tot = pd.DataFrame()

    for replicate in tqdm(range(glb['replicates'])): # for the given number of replicates
            model = IBM(glb, spc) 
            model.C_W = p.glb['C_W'] # set the exposure concentration

            model.datacollector.collect(model) # always record the initial state
            for _ in range(tmax_steps): # for the given number of model steps
                model.step() # execute a model timestep

            mout = model.datacollector.get_model_vars_dataframe().reset_index() # collect model-level state variables 
            if glb['collect_agent_data']:
                aout = model.datacollector.get_agent_vars_dataframe().reset_index() # collect individual-level state variables
            else:
                aout = pd.DataFrame()

            # add identifier columns to output dataframe
            mout['replicate'] = replicate
            mout['C_W'] = p.glb['C_W']
            aout['replicate'] = replicate
            aout['C_W'] = p.glb['C_W']

            mout_tot = pd.concat([mout_tot, mout])
            aout_tot = pd.concat([aout_tot, aout])

    return IBMOut(mout_tot, aout_tot)

def exposure_DEBIBM(p, C_Wvec):
    """
    Run the IBM with parameters `p` containing global parameters `glb` and animal parameters `spc`, 
    iterating over constant exposure concentrations `C_Wvec`.
    Returns output as named tuple of dataframes.
    """

    mout_tot = pd.DataFrame()
    aout_tot = pd.DataFrame()

    for C_W in C_Wvec: # for every concentration
        p.glb['C_W'] = C_W # update entry in the global parameters
        sim = simulate_DEBIBM(p) # run the simulation

        mout_tot = pd.concat([mout_tot, sim.mout]) # collect results

    return IBMOut(mout = mout_tot, aout = aout_tot) # return collected results
