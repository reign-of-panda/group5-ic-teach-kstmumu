# -*- coding: utf-8 -*-
# Made by NathanvEs - 15/10/2021

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit

# Change this path to whatever it is on your personal computer
folder_path = "/Users/raymondvanes/Downloads"
file_name = "/total_dataset.csv"
file_path = folder_path + file_name
dF = pd.read_csv(file_path)

# Here are all the columns
#print("All the data columns: \n")
#print(dF.columns[56:60])
#print(dF.head)

# Probalibities
mu_plus_ProbNNmu = dF['mu_plus_MC15TuneV1_ProbNNmu']
mu_minus_ProbNNmu = dF['mu_minus_MC15TuneV1_ProbNNmu']
mu_plus_ProbNNp = dF["mu_plus_MC15TuneV1_ProbNNp"]
mu_plus_probNNk = dF["mu_plus_MC15TuneV1_ProbNNk"]
mu_plus_ProbNNpi = dF["mu_plus_MC15TuneV1_ProbNNpi"]
K_ProbNNp = dF["K_MC15TuneV1_ProbNNp"]
K_probNNk = dF["K_MC15TuneV1_ProbNNk"]
K_ProbNNpi = dF["K_MC15TuneV1_ProbNNpi"]
Pi_ProbNNp = dF["Pi_MC15TuneV1_ProbNNp"]
Pi_probNNk = dF["Pi_MC15TuneV1_ProbNNk"]
Pi_ProbNNpi = dF["Pi_MC15TuneV1_ProbNNpi"]

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

B0_MM = dF["B0_MM"]
J_psi_MM = dF["J_psi_MM"]


### PARTICLE MASS CALCULATION

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

# Some plotting
#plt.hist(dF['B0_MM'], range = [5170, 5600], bins=300, density=True)
#plt.hist(dF['B0_MM'], bins=30, density=True)
#plt.show()


### SELECTION CRITERIA

def apply_selection_threshold(dataF, column, threshold, opposite=False):
    mask = (dataF[column] >= threshold)
    if opposite == True:
        dataF = dataF[~mask]
    else:
        dataF = dataF[mask]
    return dataF

"""
Some selection criteria
P(kaon): ProbNNK · (1 − ProbNNp) > 0.05 
P(pion): ProbNNπ · (1 − ProbNNK) · (1 − ProbNNp) > 0.1
"""

dF['accept_kaon'] = K_probNNk * (1 - K_ProbNNp)
dF['accept_pion'] = Pi_ProbNNpi * (1 - Pi_probNNk) * (1 - Pi_ProbNNp)
dF['accept_muon'] = dF[['mu_plus_MC15TuneV1_ProbNNmu', 'mu_plus_MC15TuneV1_ProbNNmu']].max(axis=1)
dF['dilepton_mass'] = Mass(dF['mu_minus_PE'],dF['mu_minus_P']) + Mass(dF['mu_plus_PE'],dF['mu_plus_P'])
print(mu_minus_M[1:1000:5])
print(mu_plus_M[1:1000:5])
print(dF['dilepton_mass'][1:1000:5])

dF_unfiltered = dF

# Probability selections (based on CERN paper)
dF = apply_selection_threshold(dF_unfiltered, 'accept_kaon', 0.05)
dF = apply_selection_threshold(dF, 'accept_pion', 0.1)
dF = apply_selection_threshold(dF, 'accept_muon', 0.2)
dF_filtered_out = apply_selection_threshold(dF_unfiltered, 'accept_kaon', 0.05, opposite=True)
dF_filtered_out = pd.concat([dF_filtered_out, apply_selection_threshold(dF_unfiltered, 'accept_pion', 0.1, opposite=True)], ignore_index=True).drop_duplicates()
dF_filtered_out = pd.concat([dF_filtered_out, apply_selection_threshold(dF_unfiltered, 'accept_muon', 0.2, opposite=True)], ignore_index=True).drop_duplicates()

# Transverse momenta selections (based on CERN paper)
dF = apply_selection_threshold(dF, 'mu_plus_PT', 800)
dF = apply_selection_threshold(dF, 'mu_minus_PT', 800)
dF = apply_selection_threshold(dF, 'K_PT', 250)
dF = apply_selection_threshold(dF, 'Pi_PT', 250)
dF_filtered_out = pd.concat([dF_filtered_out, apply_selection_threshold(dF_unfiltered, 'mu_plus_PT', 800, opposite=True)], ignore_index=True).drop_duplicates()
dF_filtered_out = pd.concat([dF_filtered_out, apply_selection_threshold(dF_unfiltered, 'mu_minus_PT', 800, opposite=True)], ignore_index=True).drop_duplicates()
dF_filtered_out = pd.concat([dF_filtered_out, apply_selection_threshold(dF_unfiltered, 'K_PT', 250, opposite=True)], ignore_index=True).drop_duplicates()
dF_filtered_out = pd.concat([dF_filtered_out, apply_selection_threshold(dF_unfiltered, 'Pi_PT', 250, opposite=True)], ignore_index=True).drop_duplicates()

print(len(dF),len(dF_unfiltered),len(dF_filtered_out))

# Masses selection (based on CERN paper)
# Seems already included unless I am mistaken

# K, Pi, mu: chi_IP^2 > 9 definitely already included in the data not needed to select on it
# Bo: DIRA > 0.9995 already included in the data
# Bo: chi_IP^2 < 25 already included in the data (actually < 16)


# Some plotting
plt.hist(dF_unfiltered['B0_MM'], range=[5170, 5600], bins=300, zorder=1)
plt.hist(dF['B0_MM'], range=[5170, 5600], bins=300, zorder=3)
plt.hist(dF_filtered_out['B0_MM'], range=[5170, 5600], bins=300, zorder=2)
plt.title('Invariant Mass of $B_0$ with & without background')
plt.xlabel('MM($B_0$)(MeV/$c^2$)')
plt.ylabel('Number of Candidates')
plt.legend(['unfiltered','filtered','filtered out'])
plt.show()
plt.show()

### ANGULAR DISTRIBUTION FUNCTIONS

def dgamma_dcosthetak(fl, cos_theta_k):
    """
    Returns the pdf defined above
    :param fl: F_L observable
    :param cos_theta_k: cos(theta_k)
    :return:
    """
    ctk = cos_theta_k
    acceptance = 0.5  # acceptance "function"
    scalar_array = ((3/2)*fl*ctk**2 + (3/4)*(1-fl)*(1-ctk**2)) * acceptance
    normalised_scalar_array = scalar_array * 2  # normalising scalar array to account for the non-unity acceptance function
    return normalised_scalar_array


def ll_dG_dctk(fl, _bin):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    _bin = int(_bin)
    data = bins[_bin]
    ctk = data['costhetak']
    normalised_scalar_array = dgamma_dcosthetak(fl=fl, cos_theta_k=ctk)
    return - np.sum(np.log(normalised_scalar_array))


def dgamma_dcosthetal(fl, afb, cos_theta_l):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    ctl = cos_theta_l
    acceptance = 0.5  # acceptance "function"
    scalar_array = ((3/4)*fl*(1-ctl**2) + (3/8)*(1-fl)*(1+ctl**2) + afb*ctl) * acceptance
    normalised_scalar_array = scalar_array * 2  # normalising scalar array to account for the non-unity acceptance function
    return normalised_scalar_array


def ll_dG_dctl(fl, afb, _bin):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    _bin = int(_bin)
    data = bins[_bin]
    ctl = data['costhetal']
    normalised_scalar_array = dgamma_dcosthetal(fl=fl, afb=afb, cos_theta_l=ctl)
    return - np.sum(np.log(normalised_scalar_array))


def dgamma_dphi(fl, at, aim, phi):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param at: a_t observable
    :param aim: a_im observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    acceptance = 0.5  # acceptance "function"
    scalar_array = (1/(2*np.pi)) * (1 + (1/2)*(1-fl)*at*np.cos(2*phi) + aim*np.sin(2*phi)) * acceptance
    normalised_scalar_array = scalar_array * 2  # normalising scalar array to account for the non-unity acceptance function
    return normalised_scalar_array


def ll_dG_dphi(fl, at, aim, _bin):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    _bin = int(_bin)
    data = bins[_bin]
    phi = data['phi']
    normalised_scalar_array = dgamma_dphi(fl=fl, at=at, aim=aim, phi=phi)
    return - np.sum(np.log(normalised_scalar_array))


"""
First thing to do is to split up total data in bins based on q^2 range. 
Then for each bin get the values of fl, afb, at, aim and compare to sm values.
"""


### SEPARATING DATA INTO q^2 BINS

bins = []
q_ranges = [[0.01,0.98],[1.1,2.5],[2.5,4.0],[4.0,6.0],[6.0,8.0],
           [15.0,17.0],[17.0,19.0],[11.0,12.5],[1.0,6.0],[15.0,17.9]]
q_ranges_paper = [[0.01,2.0],[2.0,4.0],[4.0,8.5],[10.0,13.0],[14.5,16.0],[16.0,23.0]]

for q_range in q_ranges_paper:
    mask = (dF['q2'] > q_range[0]) & (dF['q2'] < q_range[1])
    bin = dF[mask]
    bins.append(bin)

"""
_test_bin = 3
_test_afb = 0.7
_test_fl = 0.0

x = np.linspace(-1, 1, 500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.plot(x, [ll_dG_dctl(fl=i, afb=_test_afb, _bin=_test_bin) for i in x])
ax1.set_title(r'$A_{FB}$ = ' + str(_test_afb))
ax1.set_xlabel(r'$F_L$')
ax1.set_ylabel(r'$-\mathcal{L}$')
ax1.grid()
ax2.plot(x, [ll_dG_dctl(fl=_test_fl, afb=i, _bin=_test_bin) for i in x])
ax2.set_title(r'$F_{L}$ = ' + str(_test_fl))
ax2.set_xlabel(r'$A_{FB}$')
ax2.set_ylabel(r'$-\mathcal{L}$')
ax2.grid()
plt.tight_layout()
plt.show()
"""


### FITTING FOR THE OBSERVABLES PER q^2 BIN

ll_dG_dctl.errordef = Minuit.LIKELIHOOD
ll_dG_dctk.errordef = Minuit.LIKELIHOOD
ll_dG_dphi.errordef = Minuit.LIKELIHOOD
decimal_places = 3
starting_point = [-0.1,0.0]
fls_l, fl_errs_l, fls_k, fl_errs_k, fls_p, fl_errs_p = [], [], [], [], [], []
afbs, afb_errs = [], []
ats, at_errs = [], []
aims, aim_errs = [], []
for i in range(len(bins)):
    l = Minuit(ll_dG_dctl, fl=starting_point[0], afb=starting_point[1], _bin=int(i))
    k = Minuit(ll_dG_dctk, fl=starting_point[0], _bin=int(i))
    p = Minuit(ll_dG_dphi, fl=starting_point[0], at=starting_point[0], aim=starting_point[0], _bin=int(i))

    l.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
    l.limits=((-1.0, 1.0), (-1.0, 1.0), None)
    l.migrad()
    l.hesse()

    k.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
    k.limits = ((-1.0, 1.0), None)
    k.migrad()
    k.hesse()

    p.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
    p.limits = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), None)
    p.migrad()
    p.hesse()

    # Append all the values
    fls_l.append(l.values[0])
    afbs.append(l.values[1])
    fl_errs_l.append(l.errors[0])
    afb_errs.append(l.errors[1])

    fls_k.append(k.values[0])
    fl_errs_k.append(k.errors[0])

    fls_p.append(p.values[0])
    ats.append(p.values[1])
    aims.append(p.values[2])
    fl_errs_p.append(p.errors[0])
    at_errs.append(p.errors[1])
    aim_errs.append(p.errors[2])

    #print(f"Bin {i}: {np.round(fls_l[i], decimal_places)} pm {np.round(fl_errs_l[i], decimal_places)},",
    #      f"{np.round(afbs_l[i], decimal_places)} pm {np.round(afb_errs_l[i], decimal_places)}. "
    #      f"Function minimum considered valid: {m.fmin.is_valid}")



### PLOTTING OBSERVABLES
plotting_bool = True
plt.rcParams.update({'errorbar.capsize': 2, 'text.usetex': True})

if plotting_bool == True:
    plt.figure()
    plt.title("Values of $F_{L}$ for all 3 distributions (l, k \& phi) per $q^2$ bin")
    plt.errorbar(range(0,len(bins)),fls_p, yerr=fl_errs_p, marker='x', markersize=6, linestyle='none')
    plt.errorbar(range(0,len(bins)),fls_l, yerr=fl_errs_l, marker='x', markersize=6, linestyle='none')
    plt.errorbar(range(0,len(bins)),fls_k, yerr=fl_errs_k, marker='x', markersize=6, linestyle='none')
    plt.legend(['$F_{Lphi}$','$F_{Ll}$','$F_{Lk}$'])
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Values of $A_{FB}$ per $q^2$ bin")
    plt.errorbar(range(0,len(bins)),afbs, yerr=afb_errs, marker='x', markersize=6, linestyle='none')
    plt.legend(['$A_{FB}$'])
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Values of $A_{T}$ per $q^2$ bin")
    plt.errorbar(range(0,len(bins)),ats, yerr=at_errs, marker='x', markersize=6, linestyle='none')
    plt.legend(['$A_{T}$'])
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Values of $A_{im}$ per $q^2$ bin")
    plt.errorbar(range(0,len(bins)),aims, yerr=aim_errs, marker='x', markersize=6, linestyle='none')
    plt.legend(['$A_{im}$'])
    plt.grid()
    plt.show()

