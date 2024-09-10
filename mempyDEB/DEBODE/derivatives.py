# derivatives.py
# definition of derivative functions which make up the core of the DEB-TKTD model

import numpy as np
import warnings # throwing warnings

def LL2(x, p): 
    """
    Two-parameter log-logistic function
    """
    return 1/(1 + np.power(x/p[0], p[1]))

def LL2h(x, p):
    """
    Cumulative hazard function of the two-parameter log-logistic function, 
    used to model lethal effects under SD.
    """
    return -np.log(LL2(x, p))

def LL2M(x, p):
    """
    Cumulative hazard function of the two-parameter log-logistic function, shifted to have a lower limit of 1. 
    Used to model sublethal effects of PMoAs for which the affected state variable 
    increases with increasing damage (maintenance costs).
    """
    return 1 - np.log(LL2(x, p))

def DEBBase(t, y, glb, spc, LS_max):
    """
    DEBBase(t, y, glb, spc)

    Derivatives of the "DEBBase" model. <br>
    DEBBase is a formulation of DEBkiss with maturity, where structure is expressed as mass (instead of volume). <br>
    The TKTD part assumes log-logistic relationships between scaled damage and the relative response. <br>
    There is no explicit representation of "stress". Instead, we compute the relative response directly by applying the appropriate form of the dose-response function.
    This is the same model formulation as used in the Julia package DEBBase.jl.

    args:

    - t: current time point
    - y: vector of states
    - glb: global parameters
    - spc: species-specific parameters
    - LS_max: maximum structural length (expressed as cubic root of mass), calculated from parameters in spc.
    """

    S, R, X_emb, X, D_j, D_h = y
    
    X_emb = np.maximum(0, X_emb)
    S = np.maximum(0, S)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        LS = S**(1/3) # current structural length

        # relative responses for sublethal effects
        y_G = (int(spc['pmoa'] == 'G') * LL2(D_j, (spc['ED50_j'], spc['beta_j']))) + (int(spc['pmoa'] != 'G') * 1)
        y_M = (int(spc['pmoa'] == 'M') * LL2M(D_j, (spc['ED50_j'], spc['beta_j']))) + (int(spc['pmoa'] != 'M') * 1)
        y_A =  (int(spc['pmoa'] == 'A') * LL2(D_j, (spc['ED50_j'], spc['beta_j']))) + (int(spc['pmoa'] != 'A') * 1)
        y_R =  (int(spc['pmoa'] == 'R') * LL2(D_j, (spc['ED50_j'], spc['beta_j']))) + (int(spc['pmoa'] != 'R') * 1)

        X_emb = np.maximum(X_emb, 0)
        X = np.maximum(X, 0)
        
        if X_emb > 0: # feeeding and assimilation for embryos
            Idot = spc["Idot_max_rel_emb"] * S**(2/3)
            Adot = Idot * spc['eta_IA']
            Xdot_emb = -Idot
            Xdot = 0
            DDot_j = 0 # no change in damage for embryos
        else: # feeding, assimilation for all other life stages
            X_V = X/glb['V_patch'] 
            f = X_V / (X_V + spc['K_X'])
            Idot = f * spc["Idot_max_rel"] * S**(2/3)
            Adot = Idot * spc['eta_IA'] * y_A
            Xdot = glb['Xdot_in'] - Idot
            Xdot_emb = 0

        Mdot = y_M * spc['k_M'] * S
        Sdot = y_G * spc['eta_AS'] * (spc['kappa'] * Adot - Mdot)

        if Sdot < 0:
            Sdot = -(Mdot / spc['eta_SA'] - spc['kappa'] * Adot)
        if (S >= spc["S_p"]):
            Rdot = y_R * spc['eta_AR'] * (1 - spc['kappa']) * Adot
        else:
            Rdot = 0

        # TK for embryos
        if X_emb > 0:
            DDot_j = 0
            DDot_h = 0
        else:
            DDot_j = (spc['kD_j'] * (LS_max / LS) * (glb['C_W'] - D_j)) - (D_j * (1/S) * Sdot)
            DDot_h = (spc['kD_h'] * (LS_max / LS) * (glb['C_W'] - D_h)) - (D_h * (1/S) * Sdot)

        return Sdot, Rdot, Xdot_emb, Xdot, DDot_j, DDot_h
            