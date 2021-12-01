# -*- coding: utf-8 -*-
"""
Investigating peaking backgrounds
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Change this path to whatever it is on your personal computer (make sure you have csv files)
folder_path = "D:\OneDrive - Imperial College London\Imperial College London\Module Content\Year 3\Problem Solving\year3-problem-solving\year3-problem-solving\csv"
file_name = "/total_dataset.csv"
file_path = folder_path + file_name

dF = pd.read_csv(file_path)

# Here are all the columns from total_dataset
print(dF.columns)

#%%
"""
Some functions that'll be needed
"""

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
file_background = folder_path + "/phimumu.csv"
dF_background = pd.read_csv(file_background)

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

    return Phi_M, B0s_M

Phi_M, B0s_M = phimumu(dF)
Phi_M_back, B0s_M_back = phimumu(dF_background)

background_B0, background_Kstar = dF_background['B0_MM'], dF_background['Kstar_MM']

# Plotting for decay mode corresponding to phimumu file

# B0 mass and B0s mass (based on Kstar reconstruction)
plt.figure(figsize=(10,8))
plt.hist(B0s_M, bins = 200,histtype='step', density = False, label = 'Mass of B0s if we knew a Kaon was mistook for Pion')
# plt.hist(dF['B0_MM'], bins = 200, density = True, label = 'B0_MM from Data')
plt.hist(B0s_M_back, bins = 200, histtype='step', density = False, label = 'B0_MM from background (B0s mass)')
plt.xlim(4600,7800)
# plt.ylim(0,0.006)
#plt.hist(dF['B0_MM'], bins = 30,density = True,histtype='step',label = 'B0 Masses')
plt.tick_params(axis='both',labelsize=15)
plt.grid(True)
plt.xlabel('Mass (Mev/c^2)',fontsize=18)
plt.legend(fontsize=16)
plt.show()

# Kstar mass and phi mass (based on pion reconstruction)
plt.figure(figsize=(10,8))
plt.tick_params(axis='both',labelsize=15)
plt.grid(True)
plt.xlabel('Mass (Mev/c^2)',fontsize=18)
# plt.hist(dF['Kstar_MM'], bins = 200,density = True, histtype='step', label = 'Kstar_MM from Data')
plt.hist(Phi_M, bins = 200, density = True, histtype='step', label = 'Mass of Phi if we knew a Kaon was mistook for Pion')
plt.hist(Phi_M_back, bins = 200, histtype='step', density = True, label = 'Kstar from background (phi mass)')
#plt.hist(dF['Kstar_MM'], bins = 30,histtype='step',density = True,label = 'Kaon Masses')
plt.xlim(800,2000)
plt.ylim(0,0.004)
plt.legend(fontsize=16)
plt.show()

#%%
file_background = folder_path + "/k_pi_swap.csv"
dF_background = pd.read_csv(file_background)

def k_pi_swap(DF):
    """
    Signal decay with kaon reconstructed as pion
    """
    mu_plus_PE = dF["mu_plus_PE"]
    mu_minus_PE = dF["mu_minus_PE"]
    # K_PE = dF["K_PE"]
    # Pi_PE = dF["Pi_PE"]
    
    K_newE = reconstruct("Pi", "K", known_M['K'], DF)
    Pi_newE = reconstruct("K", "Pi", known_M['Pi'], DF)
    
    Kstar_E = K_newE + Pi_newE
    Kstar_PX = DF['K_PX'] + DF['Pi_PX']
    Kstar_PY = DF['K_PY'] + DF['Pi_PY']
    Kstar_PZ = DF['K_PZ'] + DF['Pi_PZ']
    Kstar_P = np.sqrt(Kstar_PX**2 + Kstar_PY**2 + Kstar_PZ**2)
    
    Kstar_M = Mass(Kstar_E, Kstar_P)
    
    B0_E = Kstar_E + mu_minus_PE + mu_plus_PE
    B0_PX = Kstar_PX + DF['mu_minus_PX'] + DF['mu_plus_PX']
    B0_PY = Kstar_PY + DF['mu_minus_PY'] + DF['mu_plus_PY']
    B0_PZ = Kstar_PZ + DF['mu_minus_PZ'] + DF['mu_plus_PZ']
    B0_P = np.sqrt(B0_PX**2 + B0_PY**2 + B0_PZ**2)

    B0_M = Mass(B0_E, B0_P)
    
    return Kstar_M, B0_M

Kstar_M, B0_M = k_pi_swap(dF)
Kstar_M_b, B0_M_b = k_pi_swap(dF_background)


plt.hist(B0_M, bins = 200, histtype='step', density = True, label = 'B0 mass after swapping Pi and K')
# plt.hist(dF['B0_MM'], bins = 200, histtype='step', density = True, label = 'B0_MM from total dataset')
plt.hist(B0_M_b, bins = 200, histtype='step', density = True, label = 'B0 background reconstructed')
plt.legend()
plt.show()

plt.hist(Kstar_M, bins = 200, histtype='step', density = True, label = 'Kstar mass after swapping Pi and K')
# plt.hist(dF['Kstar_MM'], bins = 200, histtype='step', density = True, label = 'Kstar_MM from total dataset')
plt.hist(Kstar_M_b, bins = 200, histtype='step', density = True, label = 'Kstar background reconstructed')
plt.legend()
plt.show()

#%%
file_background = folder_path + "/jpsi_mu_pi_swap.csv"
dF_background = pd.read_csv(file_background)

def jpsi_mu_pi_swap(DF, polarity):
    if polarity == -1:
        """
        Then we think that the mu minus was mistaken for a pion, and the pion from Kstar
        was mistaken for a mu plus.
        
        CURSES, NEEDS TO BE PI momentum, NOT MU
        """
        mu_minus_new_E = reconstruct("Pi", "mu_minus", known_M['mu'], DF)
        Pi_new_E = reconstruct("mu_plus", "Pi", known_M['Pi'], DF)
        
        j_psi_E = mu_minus_new_E + DF['mu_plus_PE']
        Kstar_E = Pi_new_E + DF['K_PE']
        
        j_psi_PX = DF['Pi_PX'] + DF['mu_plus_PX']
        j_psi_PY = DF['Pi_PY'] + DF['mu_plus_PY']
        j_psi_PZ = DF['Pi_PZ'] + DF['mu_plus_PZ']
        j_psi_P = np.sqrt(j_psi_PX**2 + j_psi_PY**2 + j_psi_PZ**2)
        j_psi_M = Mass(j_psi_E, j_psi_P)
        
        Kstar_PX = DF['K_PX'] + DF['mu_minus_PX']
        Kstar_PY = DF['K_PY'] + DF['mu_minus_PY']
        Kstar_PZ = DF['K_PZ'] + DF['mu_minus_PZ']
        Kstar_P = np.sqrt(Kstar_PX**2 + Kstar_PY**2 + Kstar_PZ**2)
        Kstar_M = Mass(Kstar_E, Kstar_P)
        
        B0_E = j_psi_E + Kstar_E
        B0_PX = Kstar_PX + j_psi_PX
        B0_PY = Kstar_PY + j_psi_PY
        B0_PZ = Kstar_PZ + j_psi_PY
        B0_P = np.sqrt(B0_PX**2 + B0_PY**2 + B0_PZ**2)

        B0_M = Mass(B0_E, B0_P)
        
    elif polarity == 1:
        mu_plus_new_E = reconstruct("Pi", "mu_plus", known_M['mu'], DF)
        Pi_new_E = reconstruct("mu_minus", "Pi", known_M['Pi'], DF)
        
        j_psi_E = mu_plus_new_E + DF['mu_minus_PE']
        Kstar_E = Pi_new_E + DF['K_PE']
        
        j_psi_PX = DF['mu_minus_PX'] + DF['Pi_PX']
        j_psi_PY = DF['mu_minus_PY'] + DF['Pi_PY']
        j_psi_PZ = DF['mu_minus_PZ'] + DF['Pi_PZ']
        j_psi_P = np.sqrt(j_psi_PX**2 + j_psi_PY**2 + j_psi_PZ**2)
        j_psi_M = Mass(j_psi_E, j_psi_P)
        
        Kstar_PX = DF['K_PX'] + DF['mu_plus_PX']
        Kstar_PY = DF['K_PY'] + DF['mu_plus_PY']
        Kstar_PZ = DF['K_PZ'] + DF['mu_plus_PZ']
        Kstar_P = np.sqrt(Kstar_PX**2 + Kstar_PY**2 + Kstar_PZ**2)
        Kstar_M = Mass(Kstar_E, Kstar_P)
        
        B0_E = j_psi_E + Kstar_E
        B0_PX = Kstar_PX + j_psi_PX
        B0_PY = Kstar_PY + j_psi_PY
        B0_PZ = Kstar_PZ + j_psi_PY
        B0_P = np.sqrt(B0_PX**2 + B0_PY**2 + B0_PZ**2)

        B0_M = Mass(B0_E, B0_P)
        
    return j_psi_M, Kstar_M, B0_M

polarity = 1

j_psi_M, Kstar_M, B0_M = jpsi_mu_pi_swap(dF, polarity)

j_psi_M_b, Kstar_M_b, B0_M_b = jpsi_mu_pi_swap(dF_background, polarity)

plt.hist(B0_M, bins = 200, histtype='step', density = True, label = 'B0 mass after swapping mu and Pi')
# plt.hist(dF['B0_MM'], bins = 200, histtype='step', density = True, label = 'B0_MM from total dataset')
plt.hist(B0_M_b, bins = 200, histtype='step', density = True, label = 'B0 background reconstructed')
# plt.xlim(4000, 100000)
plt.legend()
plt.show()

plt.hist(Kstar_M, bins = 200, histtype='step', density = True, label = 'Kstar mass after swapping mu and Pi')
# plt.hist(dF['J_psi_MM'], bins = 200, histtype='step', density = True, label = 'Kstar_MM from total dataset')
plt.hist(Kstar_M_b, bins = 200, histtype='step', density = True, label = 'Kstar background reconstructed')
plt.legend()
plt.show()  
        
plt.hist(j_psi_M, bins = 200, histtype='step', density = True, label = 'J_psi mass after swapping mu and Pi')
# plt.hist(dF['J_psi_MM'], bins = 200, histtype='step', density = True, label = 'Kstar_MM from total dataset')
plt.hist(j_psi_M_b, bins = 200, histtype='step', density = True, label = 'J_psi background reconstructed')
plt.legend()
plt.show()  
        
#%%
file_background = folder_path + "/jpsi_mu_k_swap.csv"
dF_background = pd.read_csv(file_background)

def jpsi_mu_k_swap(DF, polarity):
    if polarity == -1:
        """
        Polarity == -1
        means we think that the mu minus was mistaken for a kaon, and the kaon from Kstar
        was mistaken for a mu plus.
        """
        mu_minus_new_E = reconstruct("K", "mu_minus", known_M['mu'], DF)
        K_new_E = reconstruct("mu_plus", "K", known_M['K'], DF)
        
        j_psi_E = mu_minus_new_E + DF['mu_plus_PE']
        Kstar_E = K_new_E + DF['Pi_PE']
        
        j_psi_PX = DF['K_PX'] + DF['mu_plus_PX']
        j_psi_PY = DF['K_PY'] + DF['mu_plus_PY']
        j_psi_PZ = DF['K_PZ'] + DF['mu_plus_PZ']
        j_psi_P = np.sqrt(j_psi_PX**2 + j_psi_PY**2 + j_psi_PZ**2)
        j_psi_M = Mass(j_psi_E, j_psi_P)
        
        Kstar_PX = DF['Pi_PX'] + DF['mu_minus_PX']
        Kstar_PY = DF['Pi_PY'] + DF['mu_minus_PY']
        Kstar_PZ = DF['Pi_PZ'] + DF['mu_minus_PZ']
        Kstar_P = np.sqrt(Kstar_PX**2 + Kstar_PY**2 + Kstar_PZ**2)
        Kstar_M = Mass(Kstar_E, Kstar_P)
        
        B0_E = j_psi_E + Kstar_E
        B0_PX = Kstar_PX + j_psi_PX
        B0_PY = Kstar_PY + j_psi_PY
        B0_PZ = Kstar_PZ + j_psi_PY
        B0_P = np.sqrt(B0_PX**2 + B0_PY**2 + B0_PZ**2)

        B0_M = Mass(B0_E, B0_P)
        
    elif polarity == 1:
        """
        Polarity == +1
        means we think that the mu plus was mistaken for a kaon, and the kaon from Kstar
        was mistaken for a mu minus.
        """
        mu_plus_new_E = reconstruct("K", "mu_plus", known_M['mu'], DF)
        K_new_E = reconstruct("mu_minus", "K", known_M['K'], DF)
        
        j_psi_E = mu_plus_new_E + DF['mu_minus_PE']
        Kstar_E = K_new_E + DF['P_PE']
        
        j_psi_PX = DF['mu_minus_PX'] + DF['K_PX']
        j_psi_PY = DF['mu_minus_PY'] + DF['K_PY']
        j_psi_PZ = DF['mu_minus_PZ'] + DF['K_PZ']
        j_psi_P = np.sqrt(j_psi_PX**2 + j_psi_PY**2 + j_psi_PZ**2)
        j_psi_M = Mass(j_psi_E, j_psi_P)
        
        Kstar_PX = DF['Pi_PX'] + DF['mu_plus_PX']
        Kstar_PY = DF['Pi_PY'] + DF['mu_plus_PY']
        Kstar_PZ = DF['Pi_PZ'] + DF['mu_plus_PZ']
        Kstar_P = np.sqrt(Kstar_PX**2 + Kstar_PY**2 + Kstar_PZ**2)
        Kstar_M = Mass(Kstar_E, Kstar_P)
        
        B0_E = j_psi_E + Kstar_E
        B0_PX = Kstar_PX + j_psi_PX
        B0_PY = Kstar_PY + j_psi_PY
        B0_PZ = Kstar_PZ + j_psi_PY
        B0_P = np.sqrt(B0_PX**2 + B0_PY**2 + B0_PZ**2)

        B0_M = Mass(B0_E, B0_P)
        
    return j_psi_M, Kstar_M, B0_M

polarity = 1

j_psi_M, Kstar_M, B0_M = jpsi_mu_pi_swap(dF, polarity)

j_psi_M_b, Kstar_M_b, B0_M_b = jpsi_mu_pi_swap(dF_background, polarity)

plt.hist(B0_M, bins = 200, histtype='step', density = True, label = 'B0 mass after swapping mu and K')
# plt.hist(dF['B0_MM'], bins = 200, histtype='step', density = True, label = 'B0_MM from total dataset')
plt.hist(B0_M_b, bins = 200, histtype='step', density = True, label = 'B0 background reconstructed')
# plt.xlim(4000, 100000)
plt.legend()
plt.show()

plt.hist(Kstar_M, bins = 200, histtype='step', density = True, label = 'Kstar mass after swapping mu and K')
# plt.hist(dF['J_psi_MM'], bins = 200, histtype='step', density = True, label = 'Kstar_MM from total dataset')
plt.hist(Kstar_M_b, bins = 200, histtype='step', density = True, label = 'Kstar background reconstructed')
plt.legend()
plt.show()  
        
plt.hist(j_psi_M, bins = 200, histtype='step', density = True, label = 'J_psi mass after swapping mu and K')
# plt.hist(dF['J_psi_MM'], bins = 200, histtype='step', density = True, label = 'Kstar_MM from total dataset')
plt.hist(j_psi_M_b, bins = 200, histtype='step', density = True, label = 'J_psi background reconstructed')
plt.legend()
plt.show()  

#%%
file_background = folder_path + "/pKmumu_piTop.csv"
dF_background = pd.read_csv(file_background)

def pKmumu_piTop(DF):
    K_PE = DF["K_PE"]
    mu_plus_PE = DF["mu_plus_PE"]
    mu_minus_PE = DF["mu_minus_PE"]
        
    p_new_E = reconstruct('Pi', 'p', known_M['p'], DF) 
    
    # proton calculations
    p_PX = DF['Pi_PX']
    p_PY = DF['Pi_PY']
    p_PZ = DF['Pi_PZ']
    p_P = np.sqrt(p_PX**2 + p_PY**2 + p_PY**2)
    
    p_M = Mass(p_new_E, p_P)
    
    # bottom lambda calculations
    lambda_E = p_new_E + K_PE + mu_minus_PE + mu_plus_PE
    lambda_PX = p_PX + DF['K_PX'] + DF['mu_minus_PX'] + DF['mu_plus_PX']
    lambda_PY = p_PY + DF['K_PY'] + DF['mu_minus_PY'] + DF['mu_plus_PY']
    lambda_PZ = p_PZ + DF['K_PZ'] + DF['mu_minus_PZ'] + DF['mu_plus_PZ']
    lambda_P = np.sqrt(lambda_PX**2 + lambda_PY**2 + lambda_PZ**2)

    lambda_m = Mass(lambda_E, lambda_P)

    return p_M, lambda_m

p_M, lambda_m = pKmumu_piTop(dF)
p_M_back, lambda_m_back = pKmumu_piTop(dF_background)


# B0 mass and lambda mass (based on Kstar reconstruction)
plt.figure(figsize=(10,8))
plt.hist(lambda_m, bins = 200,histtype='step', density = True, label = 'Mass of lambda if we knew a proton was mistaken for a pion')
plt.hist(lambda_m_back, bins = 200, histtype='step', density = True, label = 'Lambda mass - reconstructed background')
plt.tick_params(axis='both',labelsize=15)
plt.grid(True)
plt.xlabel('Mass (Mev/c^2)',fontsize=18)
plt.legend(fontsize=16)
plt.show()

# proton mass 
plt.figure(figsize=(10,8))
plt.tick_params(axis='both',labelsize=15)
plt.grid(True)
plt.xlabel('Mass (Mev/c^2)',fontsize=18)
plt.hist(p_M, bins = 200, density = True, histtype='step', label = 'Mass of proton if we knew a Kaon was mistaken for a Pion')
plt.hist(p_M_back, bins = 200, histtype='step', density = True, label = 'Reconstructed proton mass')
plt.legend(fontsize=16)
plt.show()

#%%
file_background = folder_path + "/pKmumu_piTok_kTop.csv"
dF_background = pd.read_csv(file_background)

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

    return p_M, K_M, lambda_m

p_M, K_M, lambda_m = pKmumu_piTok_kTop(dF)
p_M_b, K_M_b, lambda_m_b = pKmumu_piTok_kTop(dF_background)

plt.hist(lambda_m, bins = 200, histtype='step', density = True, label = 'lambda mass after swapping p, K, and Pi')
# plt.hist(dF['B0_MM'], bins = 200, histtype='step', density = True, label = 'B0_MM from total dataset')
plt.hist(lambda_m_b, bins = 200, histtype='step', density = True, label = 'lambda background reconstructed')
# plt.xlim(4000, 100000)
plt.legend()
plt.show()

plt.hist(K_M, bins = 200, histtype='step', density = True, label = 'K mass after swapping p, K, and Pi')
# plt.hist(dF['J_psi_MM'], bins = 200, histtype='step', density = True, label = 'Kstar_MM from total dataset')
plt.hist(K_M_b, bins = 200, histtype='step', density = True, label = 'K background reconstructed')
plt.legend()
plt.show()  
        
plt.hist(p_M, bins = 200, histtype='step', density = True, label = 'proton mass after swapping p, K, and Pi')
# plt.hist(dF['J_psi_MM'], bins = 200, histtype='step', density = True, label = 'Kstar_MM from total dataset')
plt.hist(p_M_b, bins = 200, histtype='step', density = True, label = 'proton background reconstructed')
plt.legend()
plt.show()  
