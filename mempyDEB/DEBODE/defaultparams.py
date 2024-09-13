
# defaultparams.py
# definiton of default parameters to easily get started, and as a reference for development and tests

from collections import namedtuple

Params = namedtuple("Params", "glb spc")


# default parameters for the DEBBase model (a variant of DEBkiss with TKTD)
defaultparams_DEBBase = Params(
    { # global parameters
        # Simulation parameters
        'tspan' : (0,23),
        'a_int' : 2, # initial age (including embryonic development)

        # Environmental parameters
        'V_patch':  0.05,
        'Xdot_in': 375,
        'C_W' : 0.
    },
    { # default animal parameters
        'X_emb_int': 0.675, # mass of an egg
        'eta_IA_0': 0.35, # assimilation efficiency 
        'eta_AS_0': 0.9, # growth efficency
        'eta_AR_0': 0.95, # reproduction efficiency
        'Idot_max_rel': 4,  # maximum specific ingestion rate
        'Idot_max_rel_emb': 4, # maximum specific ingestion rate for embryos
        'K_X': 0.5e3, # half-saturation constant for ingestion
        'kappa': 0.9, # somatic investment fraction
        'eta_SA': 0.9, # shrinking efficiency
        'k_M_0': 0.4, # somatic maintenance constant
        'S_p': 9, # structural mass at puberty

        # sublethal TKTD parameters 
        'kD_j' : .5, # dominant rate constant
        'ED50_j' : 1., # median effective damage
        'beta_j' : 2., # slope
        'pmoa' : 'G', # physiological mode of action

        # GUTS-SD parameters
        'kD_h' : .5, # dominant rate constant 
        'ED50_h' : 2., # median effective damage
        'beta_h' : 1. # slope
    }
)

