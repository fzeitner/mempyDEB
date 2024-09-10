
# defaultparams.py
# definiton of default parameters to easily get started, and as a reference for development and tests

glb = { # global parameters
    # Simulation parameters
    'tspan' : (0,23),
    'a_int' : 2, # initial age (including embryonic development)

    # Environmental parameters
    'V_patch':  0.05,
    'Xdot_in': 375,
    'C_W' : 0.
}

spc = { # default animal parameters
    'Idot_max_rel': 4, 
    'Idot_max_rel_emb': 4, 
    'eta_IA': 0.35,
    'K_X': 0.5e3,
    'kappa': 0.9, 
    'eta_AS': 0.9, 
    'eta_SA': 0.9, 
    'eta_AR': 0.95, 
    'k_M': 0.4, 
    'S_p': 9,
    'X_emb_int': 0.675,

    # sublethal TKTD parameters 
    'kD_j' : .5,
    'ED50_j' : 1.,
    'beta_j' : 2.,
    'pmoa' : 'G',

    # GUTS-SD parameters
    'kD_h' : .5,
    'ED50_h' : 2.,
    'beta_h' : 1.
}
