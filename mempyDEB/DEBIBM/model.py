import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(context = "notebook")
from scipy.integrate import solve_ivp
from scipy import stats
from collections import namedtuple
import mesa
from tqdm import tqdm
from pathlib import Path
import os

# file organization
path = Path(__file__)
projectdir = path.parent.parent.absolute()
plotsdir = os.path.join(projectdir, 'plots')
datadir = os.path.join(projectdir, 'data')

# Collection of dose-response functions.

def LL2(x, p): 
    """
    Two-parameter log-logistic function
    """
    return 1/(1 + np.power(x/p[0], p[1]))

def LL2h(x, p):
    """
    Cumulative hazard function of the two-parameter log-logistic function
    """
    return -np.log(LL2(x, p))

def LL2M(x, p):
    """
    Cumulative hazard function of the two-parameter log-logistic function, shifted to have a lower limit of 1.
    """
    return 1 - np.log(LL2(x, p))

'''
Default global parameters for DEB-IBM.
'''
glb = { # global parameters
    # Simulation parameters
    'tspan' : (0,365),
    'tres' : 24, # temporal resolution (timesteps per day)
    'N_0' : 10, # initial population size
    'data_collection_interval' : 3.5, # intervals at which data is collected (days)
    'collect_agent_data' : True, # whether to collect agent-level data or not
    'replicates' : 3, # run replicates of the simulation

    # Environmental parameters
    'V_patch':  0.5, # volume of a single patch
    'Xdot_in': 1250, # resource input rate
    'kX_out' : 0.1, # daily resource outflow rate
    'C_W' : 0. # chemical stressor concentration

    }

'''
Default animal parameters for DEB-BIM. This contains some additional parameters compared to the ODE version.
'''
spc = { # animal parameters
    # DEB parameters
    'cv' : 0.1, # individual variability in DEB parameters, given as coefficient of variation 
    'Idot_max_rel_mean': 5, # maximum specific ingestion rate; theoretical population average
    'eta_IA_0': 0.5, # assimilation efficiency
    'K_X': 0.5e3, # half-saturation constant for resource uptake
    'kappa': 0.9, # somatic allocation fraction
    'eta_AS_0' : 0.9, # growth effieciency (transformation from assimilates to structure)
    'eta_SA' : 0.9, # structural mobilization efficiency (transformation from structure to asismilates)
    'eta_AR_0': 0.95, # reproduction efficiency
    'k_M_0': 0.55, # somatic maintenance rate constant; theoretical population average
    'S_p_mean': 9, # structural mass at puberty; theoretical population average
    'X_emb_int_mean': 0.675, # initial mass of the vitellus (~~ dry mass of an egg); theoretical population average
    'tau_R' : 2.5, # reproduction period (days)

    # starvation parameters
    'S_rel_crit' : 0.5, # critical amount of structure, relative to historical maximum structure of the individual
    'h_starve' : 0.29, # hazard rate applied when loss of structure is critical (corresponds to 75% daily survival probability)

    # aging parameters
    'a_max_mean' : 60, # average maximum lifespan
    'a_max_cv' : 0.1, # coefficient of variation in maximum lifespans

    # sublethal TKTD parameters 
    'kD_j' : .5, # dominant rate constant
    'ED50_j' : 1., # median effective damage
    'beta_j' : 1., # DRC slope
    'pmoa' : 'R', # physiological mode of action

    # GUTS parameters
    'kD_h' : .1, # dominant rate constant
    'ED50_h' : 2, # median effective damage
    'beta_h' : 1. # DRC slope
    }

#%%

def deftruncnorm(mean, cv, l, u):
    """
    Define a truncated Normal distribution with mean mu, coefficient of variation cv, lower boundary l and upper boundary u.
    """

    sd = mean * cv
    lower = 0
    upper = np.inf
    a, b = (lower - mean) / sd, (upper - mean) / sd
    d = stats.truncnorm(a, b, loc = mean, scale = sd)
    return d


# definition of some constants

S_REL_INT = 0.01 # initial amount of structure, relative to X_emb_int
# the list of inherited attributes is very similar to, but (importantly) not identical to the keys in the animal parameter dictionary
# this difference is required by the individual variability submodel
INHERITED_ATTRIBUTES = [ 
    'cv',
    'Idot_max_rel_mean',
    'eta_IA_0',
    'K_X',
    'kappa',
    'eta_AS_0',
    'eta_SA',
    'eta_AR_0',
    'k_M_0',
    'S_p_mean',
    'X_emb_int',
    'X_emb_int_mean',
    'tau_R',

    'S_rel_crit', 
    'h_starve',
    'a_max_mean',
    'a_max_cv',

    'kD_j',
    'ED50_j',
    'beta_j',
    'pmoa',
    'kD_h',
    'ED50_h',
    'beta_h'
]


class Animal(mesa.Agent):
    """
    Definition of an animal. 
    """

    def __init__(self, unique_id, model):
        """
        Initialization of an animal.
        """
        # Pass the parameters to the parent class.
        super().__init__(unique_id, model)

        # do not put initialize_statevars in here
        # the parameters have to be initialized first, and this is context-dependent

        self.ind_var_applied = False
    
    def init_statevars(self, spc):
        """
        Initialize state variables. 
        """

        self.X_emb = self.X_emb_int # it is important that X_emb is set **before** individual variability is called
        self.individual_variability() 

        self.a = 0 # age 
        self.life_stage = 1 # everyone starts as embryo
        self.S = S_REL_INT * spc['X_emb_int_mean'] # mass of the vitellus set to initial value
        
        self.S_max_hist = self.S # historical maximum structural mass

        self.Idot = 0.0 # ingestion rate
        self.Mdot = 0.0 # somatic maintenance rate
        self.clutch_size = 0 # clutch size
        self.R = 0.0 # reproduction buffer
        self.cum_repro = 0.0 # cumulative reproductive output
        self.delta_tR = 0.0 # time since last reproductive event
        
        # attributes which can be modified by a stressor are initialized with the unstressed value
        self.eta_AR = self.eta_AR_0 # reproduction efficiency 
        self.eta_IA = self.eta_IA_0 # assimilation efficiency
        self.eta_AS = self.eta_AS_0 # growth efficiency
        self.k_M = self.k_M_0 # maintenance costs
        self.Mdot_sum = 0. # cumulative somatic maintenance costs

        self.Idot = 0
        self.Adot = 0
        self.Sdot = 0
        self.delta_tR = 0

        self.cause_of_death = 0. # cause of death is encoded with a number

        self.D_j = 0 # sublethal damage 
        self.D_h = 0 # lethal damage

        self.y_G = 1 # relative responses for different PMoAs
        self.y_M = 1
        self.y_A = 1
        self.y_R = 1
        self.h_z = 0 # GUTS hazard rate

    def individual_variability(self):
        """
        Induction of individual variability by application of a scatter value drawn from a truncated normal distribution. 
        The scatter value is currently applied to the mean values given in the parameter dictionary, and not to the scattered value of the parent. 
        As a consequence, ecological and evolutionary selection are not possible in the current model. 
        The reason therefore is that the model version which does include evolutionary selection predicts 
        rapid evolution of impossibly large daphnia (up to 20-fold increase in the maximum mass compared to the parent generation). 
        In reality, there have to be physiological constraints which prevent this trend (in the given magnitude). 
        More analysis is needed on this topic!
        """
        if self.ind_var_applied:
            # this checks for possible mishaps in the implementation
            Exception("Tried to apply individual variability twice to the same animal - this should never be the case.")
        else:
            scatter = deftruncnorm(1, self.cv, 0, np.inf).rvs() # a scatter vallue is sampled from a truncated normal distribution
            
            # application of the scatter multiplier to the relevant parameters
            # this accounts for the fact that individual variability changes maximum size, and some parameters scale proportionally to maximum size
            self.Idot_max_rel = self.Idot_max_rel_mean * scatter 
            self.S_p = self.S_p_mean * scatter
            self.X_emb_int = self.X_emb_int_mean *scatter

            self.a_max = deftruncnorm(self.a_max_mean, self.a_max_cv, 0, np.inf).rvs() # introduce variability in maximum life spans

            self.ind_var_applied = True # this is to make sure that individual variability is not accidentally applied twice to the same animal

    
    def init_parameters(self, spc):
        """
        Initialize parameters from parameter dictionaries. 
        This is relevant for the initial individuals and should only be called when the model object is initialized. 
        Otherwise inherit_parameters is the relevant method.
        """

        for attr in INHERITED_ATTRIBUTES: # iterate over inherited attributes
            setattr(self, attr, spc[attr]) # assign parameter values for attribute attr based on value given in dictionary spc


    def inherit_parameters(self, parent):
        """
        Individual-level parameters are inherited from the parent animal for all animals except the initial ones.
        """

        for attr in INHERITED_ATTRIBUTES: # for each inherited attribute
            setattr(self, attr, getattr(parent, attr)) # set the child's value to the parent's value

        self.parent = parent.unique_id # record who is the parent

    # def init_random_age()

    def reset_TD(self):
        """
        Reset TD state variables.
        """
        self.y_G = 1 # relative response with respect to growth efficiency
        self.y_M = 1 # relative response with respect to maintenance costs
        self.y_A = 1 # relative response with respect to assimilation efficiency
        self.y_R = 1 # relative response with respect to reproduction efficiency
        self.h_z = 0 # hazard rate in relation to chemica stress

    def determine_life_stage(self):
        """
        Determine life stage
        """
        if self.X_emb > 0:
            self.life_stage = 1 # embryo
        elif self.S <= self.S_p:
            self.life_stage = 2 # juvenile
        else:
            self.life_stage = 3 # adult

    def life_stage_effects(self):
        """
        Calculate effects of life stage on metabolic baseline values
        """
        # (currently no life-stage specific effects implemented)


    def TK(self):
        """
        Toxicokinetics: Calculate change in scaled damage
        """

        self.D_h = self.D_h + (self.kD_h * (self.model.C_W - self.D_h)) / self.model.tres
        self.D_j = self.D_j + (self.kD_j * (self.model.C_W - self.D_j)) / self.model.tres

    def stress(self):
        """
        Calculate responses to stress: relative response for sublethal effects and hazard rate for lethal effects
        """

        self.h_z = LL2h(self.D_h, (self.ED50_h, self.beta_h))

        # relative response per PMoA
        # note the different dose-response function for 'M'
        self.y_G = (self.pmoa == 'G') * LL2(self.D_j, (self.ED50_j, self.beta_j)) + (self.pmoa != 'G')
        self.y_M = (self.pmoa == 'M') * LL2M(self.D_j, (self.ED50_j, self.beta_j)) + (self.pmoa != 'M')
        self.y_A = (self.pmoa == 'A') * LL2(self.D_j, (self.ED50_j, self.beta_j)) + (self.pmoa != 'A')
        self.y_R = (self.pmoa == 'R') * LL2(self.D_j, (self.ED50_j, self.beta_j)) + (self.pmoa != 'R')


    def apply_effects(self): 
        """
        Apply relative responses to baseline values of state variables
        """

        self.eta_AS = self.eta_AS_0 * self.y_G
        self.k_M = self.k_M_0 * self.y_M
        self.eta_IA = self.eta_IA_0 * self.y_A
        self.eta_AR = self.eta_AR_0 * self.y_R

    def calc_Idot(self):
        """
        Calculate resource ingestion rate for juvenile and adults and uptake from vitellus for embryos.
        """

        if self.life_stage == 1: # ingestion for embryos
            self.X_V = 0
            self.f_X = 1
            self.Idot = self.Idot_max_rel * np.power(self.S, 2/3)
        else: # ingestion for juveniles and adults
            self.X_V = self.model.X / self.model.V_patch # calculate resource concentration X_V (g/L) form absolute resource biomass X
            self.f_X = self.X_V / (self.K_X + self.X_V) # calculate scaled functional response
            self.Idot = self.f_X * self.Idot_max_rel * np.power(self.S, 2/3) # calculate resource ingestion rate

    def calc_Adot(self):
        """
        Caclulate assimilation rate
        """

        self.Adot = self.Idot * self.eta_IA 

    def calc_Mdot(self):
        """
        Calculate maintenance rate
        """

        self.Mdot = self.k_M * self.S

    def calc_Sdot(self):
        """
        Calculate somatic growth rate
        """

        self.Sdot = self.eta_AS * (self.kappa * self.Adot - self.Mdot) # calculate structural growth
        if self.Sdot < 0: # if growth turns out to be negative
            self.Sdot = -(self.Mdot / self.eta_SA - self.kappa * self.Adot) # overwrite growth equation with shrinking equation

    def calc_Rdot(self): 
        """
        Calculate reproduction rate
        """

        if self.S > self.S_p:
            self.Rdot = self.eta_AR * ((1 - self.kappa) * self.Adot)
        else:
            self.Rdot = 0

    def update(self):
        """
        Update animal state variables and resource.
        """

        if self.life_stage > 1: # juveniles and adults feed on external resource
            self.model.X = np.maximum(0, self.model.X - self.Idot / self.model.tres) # update external resource biomass, decreases according to ingestion rate
        else: # embryos feed on vitellus
            self.X_emb = np.maximum(0, self.X_emb - self.Idot / self.model.tres) # update vitellus
        self.S = np.maximum(0, self.S + self.Sdot / self.model.tres) # update structure
        self.S_max_hist = np.maximum(self.S, self.S_max_hist) # update historical maximum structure
        self.R += self.Rdot / self.model.tres # update reproduction buffer

    def reproduce(self):
        """
        Produce new offspring from reproduction buffer
        """
        if self.life_stage < 3: # if we have a non-adult
            self.delta_tR = 0 # do nothing
        else: # if we have an adult
            self.delta_tR += 1 / self.model.tres # increment time since last reproduction event

            if self.delta_tR >= self.tau_R: # if the reproduction period has been exceeded
                self.clutch_size = int(np.trunc(self.R / self.X_emb_int)) # calculate the clutch size. we apply the trunc function because we can only produce whole individuals. 
                self.R -= self.clutch_size * self.X_emb_int # subtract the spent energy from the reproduction buffer

                for _ in range(self.clutch_size): # for the given number of offspring individuals
                    self.model.unique_id_count += 1
                    a = Animal(self.model.unique_id_count, self.model) # create a new animal
                    a.parent = self.unique_id # save who's the parent
                    a.inherit_parameters(self) # parameters are inerhited from parent
                    a.init_statevars(spc) # intiialize state variables
                    a.cohort = self.cohort + 1 # child is one cohort higher than the parent
                    self.model.schedule.add(a) # add animal to the population
                    self.cum_repro += 1 # update the cumulative reproduction. IMPORTANT NOTE: this tracks cum repro at the time of initializing embryos, NOT at time of hatching! (the latter is what is usually observed in experiments with daphnia)
                    self.model.num_agents += 1

    def lethal_toxicity(self): 
        """
        Determination of lethal toxicity based on GUTS-SD
        """

        if np.random.uniform() > np.exp(-self.h_z / self.model.tres):
            self.cause_of_death = 3 # death by lethal toxicity is encoded by this number
            self.model.toxicity_mortality += 1
            self.model.deathlist.append(self)
    

    def starvation_mortality(self):
        """
        Individuals experience a fixed mortality probability after they lost a certain amount of their structure.
        """

        if (self.S/self.S_max_hist) < self.S_rel_crit: # check whether the amount of structure (relative to the individual's historical maximum) falls below a critical threshold
            if np.random.uniform() > np.exp(-self.h_starve / self.model.tres): # if so, apply hazard rate h_starve
                self.cause_of_death = 2 # death by starvation is encoded by this number
                self.model.starvation_mortality += 1
                self.model.deathlist.append(self)

    def aging_mortality(self):
        """
        Individuals die when they exceed their maximum lifespan. 
        The maximum lifespan is subject to individual variability.
        """

        if self.a > self.a_max:
            self.cause_of_death = 1 # death by aging is encoded by this number
            self.model.aging_mortality += 1
            self.model.deathlist.append(self)


    def death(self):
        """
        Determine whether individual has to die in the current timestep. 
        The order in which methods are called determines which causes of death can overwrite others when multiple apply at the same time.
        """

        self.lethal_toxicity()
        self.starvation_mortality()
        self.aging_mortality()


    def step(self):
        """
        Schedule for an individual animal.
        """

        self.reset_TD()
        self.determine_life_stage()
        self.TK() # toxicokinetics
        self.stress() # calculation of stress responses
        self.apply_effects() # application of stress responses
        self.calc_Idot() # calculate ingestion rate
        self.calc_Adot() # calculate assimilation rate
        self.calc_Mdot() # calculate maintenance rate
        self.calc_Sdot() # calculate structural growth rate
        self.calc_Rdot() # calculate reproduction rate
        self.update() # update state variables
        self.reproduce() # produce offspring
        self.death() # die (maybe)

        self.a += 1/self.model.tres


# data collector functions

def get_M_tot(model):
    """
    Compute the total biomass of the population.
    """

    return np.sum([a.S + a.R for a in model.schedule.agents])


# IBM object definition

class IBM(mesa.Model):
    """
    Definition of the model (IBM) object.
    """

    def assign_params(self, p):
        """
        Assign values from dictionary to IBM object
        """
        for (key,val) in zip(p.keys(), p.values()):
            setattr(self, key, val)


    def init_statevars(self):
        """
        Initialize model-level state variables
        """
        self.num_agents = 0
        self.unique_id_count = 0
        self.t_day = 0
        self.X = 0.

        # keeping track of different causes of mortality (cumulative counts)
        self.aging_mortality = 0
        self.starvation_mortality = 0
        self.toxicity_mortality = 0

    def __init__(self, glb, spc):
        """
        Initialization of the model object.
        """

        self.init_statevars()
 
        self.schedule = mesa.time.RandomActivation(self)
        self.assign_params(glb)

        if glb['collect_agent_data'] == False:
            agent_reporterdict = {}
        else:
            agent_reporterdict = { # on the individual level
                't_day' : lambda a : a.model.t_day, # time in days
                'Idot' : 'Idot', # ingestion rate
                'Adot' : 'Adot', # assimilation rate
                'Mdot' : 'Mdot', # maintenance rate
                'S' : 'S', # structure
                'R' : 'R', # reproduction buffer
                'X_V' : 'X_V', # experienced food density
                'f_X' : 'f_X', # functional response
                'D_h' : 'D_h', # lethal damage
                'D_j' : 'D_j', # sublethal damage
                'h_z' : 'h_z', # hazard rate caused by chemical stressor
                'eta_AS' : 'eta_AS', # growth efficiency
                'k_M' : 'k_M', # maintenance rate constant
                'eta_IA' : 'eta_IA', # assimilation efficiency
                'eta_AR' : 'eta_AR', # reproduction efficiency
                'X_emb' : 'X_emb', # mass of the vitellus
                'cum_repro' : 'cum_repro', # cumulative reproductive output
                'cohort' : 'cohort', # cohort (generation), starting to count at 0 for the parental cohort
                'cause_of_death' : 'cause_of_death',
                'mcov' : lambda a : a.kappa * a.Adot - a.Mdot # maintenance coverage (difference between energy available for maintenance and actual maintenance costs)
                }

        # initialization of the animal population
        for i in range(glb['N_0']): 
            a = Animal(i, self)
            a.parent = -1 # initial animals have no parents: encoded by -1
            a.cohort = 0 # parental cohort is 0
            a.init_parameters(spc) # parameters are assigned from param dictionary 
            a.init_statevars(spc) # set initial values of state variables
            self.unique_id_count += 1 # keep track of the cumulative number of animals in the simulation
            self.schedule.add(a) # add animal to the population
            self.num_agents += 1    

        # define which model output should be collected
        self.datacollector = mesa.DataCollector(
            model_reporters = { # on the model level
                't_day' : 't_day', # time in days
                'X' : 'X', # resource biomass
                'N_tot' : 'num_agents', # the total number of animals
                'M_tot' : get_M_tot,  # the total biomass
                'aging_mortality' : 'aging_mortality',
                'starvation_mortality' : 'starvation_mortality',
                'toxicity_mortality' : 'toxicity_mortality'
                }, 
            agent_reporters = agent_reporterdict
        )

    def update_resource(self):
        """
        Calculate resource inflow and outflow rate and update biomass
        """

        Xdot_out = self.kX_out * self.X # the outflow rate depends on the current biomass (the inflow rate is constant)
        self.X = np.maximum(0, self.X + (self.Xdot_in - Xdot_out) / self.tres)

    def step(self):
        """
        Schedule for the entire IBM.
        """

        self.t_day = self.schedule.steps / self.tres # update time in days
        self.deathlist = [] # reset the list of individuals to be killed at this timestep


        self.update_resource() # update the resource (food) population
        self.schedule.step() # execute the steps for animals

        # data es recorded *after* the step is executed, but *before* agents are killed
        # recording data can cost a lot of computation time, 
        # so we only do it in the specified intervals
        if np.isclose((self.t_day % self.data_collection_interval), 0, rtol = 1e-3): 
            self.datacollector.collect(self)

        # NOTE: If a spactial grid is ever added to the model, 
        # the agents also have to be removed from the grid here
        # (in addition from being removed from the schedule)
        for a in self.deathlist: # for all animals to be killed
            self.schedule.remove(a) # remove animal from scheduler
            self.deathlist.remove(a) # remove animal from death list
            self.num_agents -= 1
