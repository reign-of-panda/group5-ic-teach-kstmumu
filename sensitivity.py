# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 20:41:59 2021

@author: ib719
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def apply_selection_threshold(dataF, column, threshold, opposite=False):
    mask = (dataF[column] >= threshold)
    if opposite == True:
        dataF = dataF[~mask]
    else:
        dataF = dataF[mask]
    return dataF

X = [0, 0.025, 0.05, 0.75, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
Y = [0, 0, 0, 0, 0, 0.05, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
Z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 700, 800, 900, 800]

def filtering(filename, frame):
    dF = pd.read_csv(filename)
    B0 = dF["B0_MM"]
    N_initial = len(B0)
    K_ProbNNp = dF["K_MC15TuneV1_ProbNNp"]
    K_probNNk = dF["K_MC15TuneV1_ProbNNk"]
    K_ProbNNpi = dF["K_MC15TuneV1_ProbNNpi"]
    Pi_ProbNNp = dF["Pi_MC15TuneV1_ProbNNp"]
    Pi_probNNk = dF["Pi_MC15TuneV1_ProbNNk"]
    Pi_ProbNNpi = dF["Pi_MC15TuneV1_ProbNNpi"]
    K0_MM = dF['Kstar_MM']
    dF['accept_kaon'] = K_probNNk * (1 - K_ProbNNp) * (1 - K_ProbNNpi)
    dF['accept_pion'] = Pi_ProbNNpi * (1 - Pi_probNNk) * (1 - Pi_ProbNNp)
    dF = apply_selection_threshold(dF, 'accept_kaon', X[frame])
    dF = apply_selection_threshold(dF, 'accept_pion', Y[frame])
    dF = apply_selection_threshold(dF, 'Kstar_MM', Z[frame])
    N_final = len(dF["B0_MM"])
    return dF, N_final/N_initial

folder_path = "/Desktop/TBPS/"
signal = "sig.csv"
phimumu = "phimumu.csv"
piTok = "pKmumu_piTok_kTop.csv"
piTop = "pKmumu_piTop.csv"

for i in range(len(X)):
    dF_phi, r_phi = filtering(folder_path + phimumu, i)
    plt.hist(dF_phi['B0_MM'], range = [5000, 5700], bins=100, zorder=1, label = 'phimumu, R = %.2f' % (r_phi))
    dF_piTop, r_piTop = filtering(folder_path + piTop, i)
    plt.hist(dF_piTop['B0_MM'], range = [5000, 5700], bins=100, zorder=1, label = 'piTop, R = %.2f' % (r_piTop))
    dF_piTok, r_piTok = filtering(folder_path + piTok, i)
    plt.hist(dF_piTok['B0_MM'], range = [5000, 5700], bins=100, zorder=1, label = 'piTok, R = %.2f' % (r_piTok))
    dF_sig, r_sig = filtering(folder_path + signal, i)
    plt.hist(dF_sig['B0_MM'], range = [5000, 5700], bins=100, zorder=1, label = 'signal, R = %.2f' % (r_sig))
    plt.xlabel('Measured Mass of B0')
    plt.ylabel('Number of Candidates')
    plt.title('(K, pi, K*MM) : (%.2f, %.2f, %d)' % (X[i], Y[i], Z[i]))
    plt.legend()
#    plt.savefig("Variation_%s.png" % (i), dpi = 250)
    plt.show()

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def apply_selection_threshold(dataF, column, threshold, opposite=False):
    mask = (dataF[column] >= threshold)
    if opposite == True:
        dataF = dataF[~mask]
    else:
        dataF = dataF[mask]
    return dataF

K = [0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.7] 
pi = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
K_MM = [700, 750, 800, 850, 900]

test1 = [K, np.zeros(len(K)), np.zeros(len(K))]
test2 = [np.zeros(len(pi)), pi, np.zeros(len(pi))]
test3 = [np.zeros(len(K_MM)), np.zeros(len(K_MM)), K_MM]
analysis = [test1, test2, test3]
title = ['K_crit', 'pi_crit', 'K*_MM_crit']

def filtering(filename, frame, parameter): 
    dF = pd.read_csv(filename)
    B0 = dF["B0_MM"]
    N_initial = len(B0)
    K_ProbNNp = dF["K_MC15TuneV1_ProbNNp"]
    K_probNNk = dF["K_MC15TuneV1_ProbNNk"]
    K_ProbNNpi = dF["K_MC15TuneV1_ProbNNpi"]
    Pi_ProbNNp = dF["Pi_MC15TuneV1_ProbNNp"]
    Pi_probNNk = dF["Pi_MC15TuneV1_ProbNNk"]
    Pi_ProbNNpi = dF["Pi_MC15TuneV1_ProbNNpi"]
    K0_MM = dF['Kstar_MM']
    dF['accept_kaon'] = K_probNNk * (1 - K_ProbNNp) * (1 - K_ProbNNpi)
    dF['accept_pion'] = Pi_ProbNNpi * (1 - Pi_probNNk) * (1 - Pi_ProbNNp)
    dF = apply_selection_threshold(dF, 'accept_kaon', parameter[0][frame]) 
    dF = apply_selection_threshold(dF, 'accept_pion', parameter[1][frame])  #Y
    dF = apply_selection_threshold(dF, 'Kstar_MM', parameter[2][frame]) #Z
    N_final = len(dF["B0_MM"])
    return dF, N_final/N_initial

folder_path = "/Desktop/TBPS/"
signal = "sig.csv"
phimumu = "phimumu.csv"
piTok = "pKmumu_piTok_kTop.csv"
piTop = "pKmumu_piTop.csv"

fig, axs = plt.subplots(3, figsize = (6, 12))
for i in range(0, 3):
    R_sig = []
    R_phi = []
    R_piTok = []
    R_piTop = []
    for j in range(len(analysis[i][i])):
        dF_piTop, r_piTop = filtering(folder_path + piTop, j, analysis[i])
        R_piTop = np.append(R_piTop, r_piTop)
        dF_piTok, r_piTok = filtering(folder_path + piTok, j, analysis[i])
        R_piTok = np.append(R_piTok, r_piTok)
        dF_phi, r_phi = filtering(folder_path + phimumu, j, analysis[i])
        R_phi = np.append(R_phi, r_phi)
        dF_sig, r_sig = filtering(folder_path + signal, j, analysis[i])
        R_sig = np.append(R_sig, r_sig)
    axs[i].plot(analysis[i][i], R_phi, color = 'blue', label = 'phi')
    axs[i].plot(analysis[i][i], R_piTok, color = 'green', label = 'piTok')
    axs[i].plot(analysis[i][i], R_piTop, color = 'orange', label = 'piTop')
    axs[i].plot(analysis[i][i], R_sig, label = 'signal', color = 'red')
    axs[i].legend()
    axs[i].title.set_text('Sensitivity due to change in %s' % (title[i]))
    axs[i].set_ylabel('Proportion remained')

plt.savefig('Sensitivity_analysis.png', dpi = 250)
plt.show()
        
        
        


                               

