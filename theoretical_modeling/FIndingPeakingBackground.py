# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy 
from scipy.stats import norm 

folder_path = r'C:\Users\user\Documents\Imperial\Year 3\Comprehensives\TBPS\box_data_files' # change to whatever ur path is otherwise it'll get lost

known_M = dict([ # in MeV
            ('Pi', 139.6),
            ('K', 493.677),
            ('Kstar', 'heckin mystery'),
            ('mu', 105.66),
            ('B0', 5279.65),
            ('B0s', 5366.88),
            ('J_psi', 3096.9),
            ('phi', 1019.461),
            ('lambda_0_b', 5619.6),
            ('p', 938.27)
            ])

def open_file(file_name): # opens the pickle file
    path = folder_path + file_name
    return pd.read_pickle(path)

def Mass(PE, P):
    """
    Returns the mass in units of MeV/c^2
    *c^2 is set to 1 because nuclear physics
    """
    return (PE**2 - P**2)**0.5

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

def jpsiKM2(DF):
    E_k2m = reconstruct('K', 'mu_plus', known_M['mu'], DF)
    E_m2k = reconstruct('mu_plus', 'K', known_M['K'], DF)
    mu_minus_PE = DF["mu_minus_PE"]
    Pi_PE = DF["Pi_PE"]
    
    jpsi_E = mu_minus_PE + E_k2m
    kstar_E = Pi_PE + E_m2k
    
    jpsi_PX = DF['mu_minus_PX'] + DF['K_PX']
    jpsi_PY = DF['mu_minus_PY'] + DF['K_PY']
    jpsi_PZ = DF['mu_minus_PZ'] + DF['K_PZ']
    jpsi_P = np.sqrt(jpsi_PX**2 + jpsi_PY**2 + jpsi_PZ**2)
    
    kstar_PX = DF['Pi_PX'] + DF['mu_plus_PX']
    kstar_PY = DF['Pi_PY'] + DF['mu_plus_PY']
    kstar_PZ = DF['Pi_PZ'] + DF['mu_plus_PZ']
    kstar_P = np.sqrt(kstar_PX**2 + kstar_PY**2 + kstar_PZ**2)
    
    jpsi_RM = Mass(jpsi_E, jpsi_P)
    kstar_RM = Mass(kstar_E, kstar_P)
    
    jpsi_MM = DF["J_psi_MM"]
    kstar_MM = DF["Kstar_MM"]
    
    jpsi_MD = jpsi_MM - jpsi_RM
    kstar_MD = kstar_MM - kstar_RM

    return jpsi_MD, kstar_MD, jpsi_RM, kstar_RM

def jpsiPM(DF):
    E_m2p = reconstruct('mu_minus', 'Pi', known_M['Pi'], DF)
    E_p2m = reconstruct('Pi', 'mu_minus', known_M['mu'], DF)
    K_PE = DF["K_PE"]
    mu_plus_PE = DF["mu_plus_PE"]
    
    jpsi_E = mu_plus_PE + E_p2m
    kstar_E = K_PE + E_m2p
    
    jpsi_PX = DF['mu_plus_PX'] + DF['Pi_PX']
    jpsi_PY = DF['mu_plus_PY'] + DF['Pi_PY']
    jpsi_PZ = DF['mu_plus_PZ'] + DF['Pi_PZ']
    jpsi_P = np.sqrt(jpsi_PX**2 + jpsi_PY**2 + jpsi_PZ**2)
    
    kstar_PX = DF['K_PX'] + DF['mu_minus_PX']
    kstar_PY = DF['K_PY'] + DF['mu_minus_PY']
    kstar_PZ = DF['K_PZ'] + DF['mu_minus_PZ']
    kstar_P = np.sqrt(kstar_PX**2 + kstar_PY**2 + kstar_PZ**2)
    
    jpsi_RM = Mass(jpsi_E, jpsi_P)
    kstar_RM = Mass(kstar_E, kstar_P)
    
    jpsi_MM = DF["J_psi_MM"]
    kstar_MM = DF["Kstar_MM"]
    
    jpsi_MD = jpsi_MM - jpsi_RM
    kstar_MD = kstar_MM - kstar_RM
    
    return jpsi_MD, kstar_MD, jpsi_RM, kstar_RM
    
    
#%% run this cell for J/Psi Mu K swap background plots
background = open_file("/jpsi_mu_k_swap.pkl")
signal = open_file("/sig.pkl")
jpsi_BD, kstar_BD, jpsi_BR, kstar_BR = jpsiKM2(background)
jpsi_SD, kstar_SD, jpsi_SR, kstar_SR = jpsiKM2(signal)

plt.figure(1)
plt.title("Reconstructed mass for J/Psi")
plt.hist(jpsi_SR, alpha = 0.7, label = "signal", bins = 30)
plt.hist(jpsi_BR, alpha = 0.7, label = "background", bins = 30)
plt.legend(loc = "upper right")
plt.xlabel("RM")

plt.figure(2)
plt.title("Reconstructed mass for Kstar")
plt.hist(kstar_SR, alpha = 0.7, label = "signal", bins = 30)
plt.hist(kstar_BR, alpha = 0.7, label = "background", bins = 30)
plt.legend(loc = "upper right")
plt.xlabel("RM")

plt.figure(3)
plt.title("Difference in measured and reconstructed mass for J/Psi")
plt.hist(jpsi_SD, alpha = 0.7, label = "signal")
plt.hist(jpsi_BD, alpha = 0.7, label = "background")
plt.legend(loc = "upper right")
plt.xlabel("MM-RM")

plt.figure(4)
plt.title("Difference in measured and reconstructed mass for Kstar")
plt.hist(kstar_SD, alpha = 0.7, label = "signal", bins = 30)
plt.hist(kstar_BD, alpha = 0.7, label = "background", bins = 30)
plt.legend(loc = "upper right")
plt.xlabel("MM-RM")

#%% run this cel for J/Psi Mu Pi swap background plots
background = open_file("/jpsi_mu_pi_swap.pkl")
signal = open_file("/sig.pkl")
jpsi_BD, kstar_BD, jpsi_BR, kstar_BR = jpsiPM(background)
jpsi_SD, kstar_SD, jpsi_SR, kstar_SR = jpsiPM(signal)

plt.figure(1)
plt.title("Reconstructed mass for J/Psi")
plt.hist(jpsi_SR, alpha = 0.7, label = "signal", bins = 30)
plt.hist(jpsi_BR, alpha = 0.7, label = "background", bins = 30)
plt.legend(loc = "upper right")
plt.xlabel("RM")

plt.figure(2)
plt.title("Reconstructed mass for Kstar")
plt.hist(kstar_SR, alpha = 0.7, label = "signal", bins = 30)
plt.hist(kstar_BR, alpha = 0.7, label = "background", bins = 30)
plt.legend(loc = "upper right")
plt.xlabel("RM")

plt.figure(3)
plt.title("Difference in measured and reconstructed mass for J/Psi")
plt.hist(jpsi_SD, alpha = 0.7, label = "signal", bins = 30)
plt.hist(jpsi_BD, alpha = 0.7, label = "background", bins = 30)
#plt.hist(jpsi_TD, label  = "total")
plt.legend(loc = "upper right")
plt.xlabel("MM-RM")


plt.figure(4)
plt.title("Difference in measured and reconstructed mass for Kstar")
plt.hist(kstar_SD, alpha = 0.7, label = "signal", bins = 30)
plt.hist(kstar_BD, alpha = 0.7, label = "background", bins = 30)
#plt.hist(kstar_TD, alpha = 0.7, label = "total")
plt.legend(loc = "upper right")
plt.xlabel("MM-RM")

