# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 12:57:16 2021

@author: zoyaa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy 
from scipy.stats import norm 

def Mass(PE, P):
    """
    Returns the mass in units of MeV/c^2
    *c^2 is set to 1 because nuclear physics
    """
    return (PE**2 - P**2)**0.5

known_M = dict([ # in MeV
            ('Pi', 139.6),
            ('K', 493.677),
            ('Kstar', 'fuckin mystery'),
            ('mu', 105.66),
            ('B0', 5279.65),
            ('B0s', 5366.88),
            ('J_psi', 3096.9),
            ('phi', 1019.461),
            ('lambda_0_b', 5619.6),
            ('p', 938.27)
            ])

def reconstruct(p1, p2, p2_M, DF):
    """
    Returns the correct calculated energy
    p1: misidentified
    p2: actual # This variable input is not used - it's just there for me to not lose track of particles
    p2_M: Mass of p2
    DF: Dataframe containing momentum of p1
    """
    p1_P = DF[p1 + "_P"]  
    p1_new_E = np.sqrt(p1_P**2 + p2_M**2) # p1 energy recalculated using p2 mass
    
    return p1_new_E
#%%
"PHI_MU_MU BACKGROUND"

def phimumu(DF):#switching Pion for Kaon
    K_PE = DF["K_PE"]
    mu_plus_PE = DF["mu_plus_PE"]
    mu_minus_PE = DF["mu_minus_PE"]
        
    E_psk = reconstruct('Pi', 'K', known_M['K'], DF) # Energy of Kaon (that was reconstrcuted as a pion)
    
    # Phi calculations
    Phi_E = K_PE + E_psk
    Phi_PX = DF['K_PX'] + DF['Pi_PX']
    Phi_PY = DF['K_PY'] + DF['Pi_PY']
    Phi_PZ = DF['K_PZ'] + DF['Pi_PZ']
    Phi_P = np.sqrt(Phi_PX**2 + Phi_PY**2 + Phi_PZ**2)
    
    Phi_M = Mass(Phi_E,Phi_P)
    
    # B0s calculations
    B0s_E = Phi_E + mu_minus_PE + mu_plus_PE
    B0s_PX = Phi_PX + DF['mu_minus_PX'] + DF['mu_plus_PX']
    B0s_PY = Phi_PY + DF['mu_minus_PY'] + DF['mu_plus_PY']
    B0s_PZ = Phi_PZ + DF['mu_minus_PZ'] + DF['mu_plus_PZ']
    B0s_P = np.sqrt(B0s_PX**2 + B0s_PY**2 + B0s_PZ**2)

    B0s_M = Mass(B0s_E,B0s_P)

    return Phi_M

"pKmumu_piTok_kTop"


def pKmumu_piTok_kTop(DF):
    mu_plus_PE = DF["mu_plus_PE"]
    mu_minus_PE = DF["mu_minus_PE"]
    p_new_E = reconstruct('K', 'p', known_M['p'], DF) 
    K_new_E = reconstruct("Pi", "K", known_M['K'], DF)
    
    # proton calculations
    p_PX = DF['K_PX']
    p_PY = DF['K_PY']
    p_PZ = DF['K_PZ']
    p_P = np.sqrt(p_PX**2 + p_PY**2 + p_PY**2)
    p_M = Mass(p_new_E, p_P)
    
    # Kaon calculations
    K_PX = DF['Pi_PX']
    K_PY = DF['Pi_PY']
    K_PZ = DF['Pi_PZ']
    K_P = np.sqrt(K_PX**2 + K_PY**2 + K_PY**2)
    K_M = Mass(K_new_E, K_P)
    
    # Bottom lambda calculaitons
    lambda_E = p_new_E + K_new_E + mu_minus_PE + mu_plus_PE
    lambda_PX = p_PX + K_PX + DF['mu_minus_PX'] + DF['mu_plus_PX']
    lambda_PY = p_PY + K_PY + DF['mu_minus_PY'] + DF['mu_plus_PY']
    lambda_PZ = p_PZ + K_PZ + DF['mu_minus_PZ'] + DF['mu_plus_PZ']
    lambda_P = np.sqrt(lambda_PX**2 + lambda_PY**2 + lambda_PZ**2)

    lambda_m = Mass(lambda_E, lambda_P)

    return lambda_m