# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 09:35:25 2021

@author: therm
"""

"""
Try some random plotting of the data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
All file names

"/acceptance_mc.csv"

jpsi_mu_k_swap
sig
phimumu
"""
# Change this path to whatever it is on your personal computer
folder_path = "D:\OneDrive - Imperial College London\Imperial College London\Module Content\Year 3\Problem Solving\year3-problem-solving\year3-problem-solving\csv"
file_name = "/total_dataset.csv"
file_path = folder_path + file_name

dF = pd.read_csv(file_path)

# Here are all the columns
print(dF.columns)

#%%

"""
Some probability conditions I was trying out

P(kaon): ProbNNK · (1 − ProbNNp) > 0.05 

P(pion): ProbNNπ · (1 − ProbNNK) · (1 − ProbNNp) > 0.1

"""
# Probalibities
mu_plus_ProbNNp = dF["mu_plus_MC15TuneV1_ProbNNp"]
mu_plus_probNNk = dF["mu_plus_MC15TuneV1_ProbNNk"]
mu_plus_ProbNNpi = dF["mu_plus_MC15TuneV1_ProbNNpi"]

accept_kaon = mu_plus_probNNk * (1 - mu_plus_ProbNNp)
accept_pion = mu_plus_ProbNNpi * (1 - mu_plus_probNNk) * (1 - mu_plus_ProbNNpi)


# Transverse Momentum
mu_plus_PT = dF["mu_plus_PT"]
mu_minus_PT = dF["mu_minus_PT"]
K_PT = dF["K_PT"]
Pi_PT = dF["Pi_PT"]

# Total momentum (Units of MeV)
mu_plus_P = dF["mu_plus_P"]
mu_minus_P = dF["mu_minus_P"]
K_P = dF["K_P"]
Pi_P = dF["Pi_P"]

# 4 vector energy
mu_plus_PE = dF["mu_plus_PE"]
mu_minus_PE = dF["mu_minus_PE"]
K_PE = dF["K_PE"]
Pi_PE = dF["Pi_PE"]

# Spatial part of momentum 4 vector
mu_plus_PX = dF["mu_plus_PX"]
mu_plus_PY = dF["mu_plus_PY"]
mu_plus_PZ = dF["mu_plus_PZ"]

mu_minus_PX = dF["mu_minus_PX"]
mu_minus_PY = dF["mu_minus_PX"]
mu_minus_PZ = dF["mu_minus_PZ"]

K_PX = dF["K_PX"]
K_PY = dF["K_PY"]
K_PZ = dF["K_PZ"]

Pi_PX = dF["Pi_PX"]
Pi_PY = dF["Pi_PY"]
Pi_PZ = dF["Pi_PZ"]

# Other data
cos_l = dF["costhetal"]
cos_k = dF["costhetak"]
q2 = dF["q2"]

# Masses
def Mass(PE, P):
    """
    Returns the mass in units of MeV/c^2
    """
    return (PE**2 - P**2)**0.5
mu_plus_M = Mass(mu_plus_PE, mu_plus_P)
mu_minus_M = Mass(mu_minus_PE, mu_minus_P)
K_M = Mass(K_PE, K_P)
Pi_M = Mass(Pi_PE, Pi_P)
total_mass = mu_plus_M + mu_minus_M + K_M + Pi_M # kaon, pion, muon+ and muon-

B0_MM = dF["B0_MM"]
J_psi_MM = dF["J_psi_MM"]


# Some plotting
plt.hist(B0_MM, bins=30, density=True)








