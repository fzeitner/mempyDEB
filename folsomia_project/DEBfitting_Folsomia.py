import pandas as pd
import numpy as np
import os
from copy import deepcopy
import warnings
import pprint 

from ModelFitting import *

from mempyDEB.DEBODE.simulators import *
from mempyDEB.DEBODE.defaultparams import *

# Konstanten

EMB_DEV_TIME = 2 # geschätzte embryonalentwicklungszeit (realistisch für Daphnia)
S_MAX_REFERENCE = 388.4 # maximale Strukturelle Masse, die durch die Anfangswerte impliziert ist


def collembola_length_to_weight(L_mm):
    """
    Length-weight-relationship for Collembola (Mercer et al. 2002, Antarctic Science). 

    params:
    - `L_mm`: Length in mm
    
    output:
    - `m_mug`: Dry mass in mug
    """
    m_mug = np.exp(1.339 + 1.992*np.log(L_mm))

    return m_mug


def calc_S_max(spc: dict):
    """
    Berchnung maximaler struktureller Masse auf Basis der DEB-Parameter.
    """ 

    return np.power((spc['kappa'] * spc['Idot_max_rel'] * spc['eta_IA_0']) / spc['k_M_0'], 3)

def load_data():
    """
    Einlesen der Kontrolldaten für Folsomia.
    """

    data = pd.read_csv('folsomia_temperature_cadmium_growth_tidy.csv', header = 5)
    data = data.loc[lambda df : df.T_cels == 20]
    data = data.loc[lambda df : df.C_F == 0]
    data = data.assign(S = collembola_length_to_weight(data.length_mm))

    return data

def plot_data(data):
    """
    Plotten der Kontrolldaten für Folsomia. 
    Gibt ein `fig, ax`-Tuple zurück.
    """

    fig, ax = plt.subplots(ncols = 1, figsize = (8,6), sharex = True)

    sns.lineplot(data, x = 't_day', y = 'S', marker = 'o', color = 'black')
    
    ax.set(xlabel = "Zeit (d)", ylabel = "Strukturelle Masse (mugC)")
    
    sns.despine()
    plt.tight_layout()

    return fig,ax

#### Simulator-Funktion

def define_simulator(f: ModelFit):

    """
    Definition der Simulator-Funktion für DEB-Kalibrierung.
    """

    def simulator(theta: dict) -> tuple: # theta = rand(priors)

        p = deepcopy(f.defaultparams)
        p.spc.update(theta) # macht das gleiche wie p.spc['Idot_max_rel'] = theta['Idot_max_rel'] für alle einträge in theta

        
        S_max_theta = calc_S_max(p.spc) # implizierte maximale Struktur auf basis von theta
        zoom_factor_theta = S_max_theta / S_MAX_REFERENCE # zoom factor auf basis von theta
        
        # diese Parameter skalieren mit der neuen maximalen Größe

        p.spc['Idot_max_rel_emb'] *= zoom_factor_theta**(1/3)
        p.spc['X_emb_int'] *= zoom_factor_theta
        p.spc['S_p'] *= zoom_factor_theta
        

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prediction = simulate_DEBBase(p).assign(
                cum_repro = lambda df : np.trunc(df.R / p.spc['X_emb_int']).shift(EMB_DEV_TIME, fill_value = 0)
                ).rename({'t' : 't_day'}, axis = 1)

            return prediction
        
    return simulator
    
def define_defaultparams():
    """
    Definition der Standard-Parameterwerte für DEB-Modellkalibrierung. 

    Modellwährung ist mugC. 
    """

    glb = {
        'C_W': 0.0,
        'V_patch': 0.05,
        'Xdot_in': 1e10,
        'a_int': 6,
        'tspan': (0, 80),
        "Cd_food": 0.
    }

    spc = {
        'X_emb_int': 19.42, 
        'eta_IA_0': 0.3333333333333333, 
        'eta_AS_0': 0.9, 
        'eta_AR_0': 0.95, 
        'Idot_max_rel': 12.256744759847304, 
        'Idot_max_rel_emb': 12.256744759847304, 
        'K_X': 500.0, 
        'kappa': 0.9, 
        'eta_SA': 0.9, 
        'k_M_0': np.float64(0.5039684199579493), 
        'S_p': 258.93333333333334, 
        'kD_j': 0.5, 
        'ED50_j': 1.0, 
        'beta_j': 2.0, 
        'pmoa': 'G', 
        'kD_h': 0.5, 
        'ED50_h': 2.0, 
        'beta_h': 1.0
        }

    p = Params(glb = glb, spc = spc)

    return p

def define_loss(constants = None):

    def loss(
        prediction: pd.DataFrame,
        data: pd.DataFrame, 
        ) -> float: # returns a scalar value

        # Skalierung der Vorhersagen
        prediction = prediction.assign(
            S_scaled = lambda df : df.S / constants['scale_factor_S'], 
            #cum_repro_scaled = lambda df : df.cum_repro / constants['scale_factor_R']
            )

        # Zusammenfassung von Vorhersagen und Daten in einen einzelnen Datensatz
        eval_df = pd.merge(prediction, data, on = 't_day', how = 'right', suffixes = ['_predicted', '_observed'])
        
        # Berechnung der Loss-Funktion
        loss_S = logMSE(eval_df.S_scaled_predicted, eval_df.S_scaled_observed)
        #loss_R = logMSE(eval_df.cum_repro_scaled_predicted, eval_df.cum_repro_scaled_observed)
        
        return loss_S # nur die komplette Loss muss zurückgegeben werden 
    
    return loss


def setup_modelfit():
    
    f = ModelFit()
    f.data = load_data()

    # Konstanten die während des Fittings genutzt werden
    
    constants = {
        'scale_factor_S' : np.max(f.data.S),
        #'scale_factor_R' : np.max(f.data.cum_repro_mean)
    }
    
    # Skalierung der Daten

    f.data = f.data.assign( 
        S_scaled = lambda df : df.S / constants['scale_factor_S'], # Berechnung der skalierten Struktur
        #cum_repro_scaled = lambda df : df.cum_repro_mean / constants['scale_factor_R'] # Berechnung der skalierten Reproduktion
        )

    # Definition von Anfangswerten der Parameter

    f.defaultparams = define_defaultparams() # enthält alle Parameter, die für die Simulationen notwendig sind

    # enthält nur die Parameter, die kalibriert werden
    # was nicht in intugess ist, wird auf den wert in defaultparams fixiert 
    
    f.intguess = { 
        'Idot_max_rel' : f.defaultparams.spc['Idot_max_rel'],
        'eta_AS_0' : f.defaultparams.spc['eta_AS_0']
        }

    f.simulator = define_simulator(f)
    f.loss = define_loss(constants)

    # define_objective_function kombiniert simulator und loss in eine einzelne Funktion
    # wenn sich eins von beiden ändert, muss auch define_ojective_function neu aufgerufen werden!
    f.define_objective_function()

    return f

def fit_model():

    f = setup_modelfit() # generiere ModelFit-Instanz
    f.run_optimization() # führe Kalibrierung durch

    print(f"Estimated parameter values: {f.p_opt}")

    # Simulation optimisierter Parameter

    p = deepcopy(f.defaultparams)
    p.spc.update(f.p_opt)  

    sim_opt = f.simulator(p.spc)

    # Visual predictive check

    fig, ax = plot_data(f.data)

    sns.lineplot(sim_opt, x = 't_day', y = 'S', label = "Modell")
    #sns.lineplot(sim_opt, x = 't_day', y = 'cum_repro', ax = ax[1])

    ax.legend()

    return f