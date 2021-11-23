# -*- coding: utf-8 -*-
# Made by NathanvEs - 15/10/2021

import numpy as np
import scipy as sp
import pandas as pd
import Acceptance
import Background
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from numpy.polynomial import legendre as L
from iminuit import Minuit

# Change this path to whatever it is on your personal computer
folder_path = "/Users/raymondvanes/Downloads"
file_name = "/total_dataset.csv"
file_path = folder_path + file_name
dF_total = pd.read_csv(file_path)
print('Reading total dataset done')

# All functions:
# Fitting to the invariant mass plot
def gauss_exp(x, a, mean, sigma, A, b):
    return a * np.exp(-((x - mean) ** 2 / (2 * sigma ** 2))) + A * np.exp(b * (x - 5170))
def gauss(x, a, mean, sigma):
    return a * np.exp(-((x - mean) ** 2 / (2 * sigma ** 2)))
def exp_tail(x, A, b):
    return A * np.exp(b * (x - 5170))

# PARTICLE MASS CALCULATION
def Mass(PE, P):
    """
    Returns the mass in units of MeV/c^2
    """
    return (PE**2 - P**2)**0.5

# Angular distribution functions
def dgamma_dcosthetak(cos_theta_k, fl):
    """
    Returns the pdf defined above
    :param fl: F_L observable
    :param cos_theta_k: cos(theta_k)
    :return:
    """
    ctk = cos_theta_k
    # print('This is ctk: ')
    # print(ctk)
    # acceptance = 0.5  # acceptance "function"
    scalar_array = ((3 / 2) * fl * ctk ** 2 + (3 / 4) * (1 - fl) * (1 - ctk ** 2)) #* L.legval(ctk, legendre_k)
    normalised_scalar_array = scalar_array * 1  # normalising scalar array to account for the non-unity acceptance function
    return normalised_scalar_array
def ll_dG_dctk(fl, data):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    print(type(data))
    ctk = data['costhetak']
    normalised_scalar_array = dgamma_dcosthetak(fl=fl, cos_theta_k=ctk)
    return - np.sum(np.log(normalised_scalar_array))
def dgamma_dcosthetal(cos_theta_l, fl, afb):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    ctl = cos_theta_l
    # print('This is ctl: ')
    # print(ctl)
    acceptance = 0.5  # acceptance "function"
    scalar_array = ((3 / 4) * fl * (1 - ctl ** 2) + (3 / 8) * (1 - fl) * (1 + ctl ** 2) + afb * ctl) #* L.legval(ctl,legendre_l)
    normalised_scalar_array = scalar_array * 1  # normalising scalar array to account for the non-unity acceptance function
    return normalised_scalar_array
def ll_dG_dctl(fl, afb, data):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    print(type(data))
    ctl = data['costhetal']
    normalised_scalar_array = dgamma_dcosthetal(fl=fl, afb=afb, cos_theta_l=ctl)
    return - np.sum(np.log(normalised_scalar_array))
def dgamma_dphi(phi, fl, at, aim):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param at: a_t observable
    :param aim: a_im observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    acceptance = 1.0  # acceptance "function"
    scalar_array = (1 / (2 * np.pi)) * (
                1 + (1 / 2) * (1 - fl) * at * np.cos(2 * phi) + aim * np.sin(2 * phi)) * acceptance
    normalised_scalar_array = scalar_array * 1  # normalising scalar array to account for the non-unity acceptance function
    return normalised_scalar_array
def ll_dG_dphi(fl, at, aim, data):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    phi = data['phi']
    normalised_scalar_array = dgamma_dphi(fl=fl, at=at, aim=aim, phi=phi)
    return - np.sum(np.log(normalised_scalar_array))

# Selection criteria
def apply_selection_threshold(dataF, column, threshold, opposite=False):
    mask = (dataF[column] >= threshold)
    if opposite == True:
        dataF = dataF[~mask]
    else:
        dataF = dataF[mask]
    return dataF

# Calculate simple probabilities
dF_total['accept_kaon'] = dF_total["K_MC15TuneV1_ProbNNk"] * (1 - dF_total["K_MC15TuneV1_ProbNNp"])
dF_total['accept_pion'] = dF_total["Pi_MC15TuneV1_ProbNNpi"] * (1 - dF_total["Pi_MC15TuneV1_ProbNNk"]) * (1 - dF_total["Pi_MC15TuneV1_ProbNNp"])
dF_total['accept_muon'] = dF_total[['mu_plus_MC15TuneV1_ProbNNmu', 'mu_plus_MC15TuneV1_ProbNNmu']].max(axis=1)

dF_unfiltered = dF_total

# Probability selections (based on CERN paper)
dF_total = apply_selection_threshold(dF_unfiltered, 'accept_kaon', 0.05)
dF_total = apply_selection_threshold(dF_total, 'accept_pion', 0.1)
dF_total = apply_selection_threshold(dF_total, 'accept_muon', 0.2)
dF_filtered_out = apply_selection_threshold(dF_unfiltered, 'accept_kaon', 0.05, opposite=True)
dF_filtered_out = pd.concat([dF_filtered_out, apply_selection_threshold(dF_unfiltered, 'accept_pion', 0.1, opposite=True)],ignore_index=True).drop_duplicates()
dF_filtered_out = pd.concat([dF_filtered_out, apply_selection_threshold(dF_unfiltered, 'accept_muon', 0.2, opposite=True)],ignore_index=True).drop_duplicates()

# Transverse momenta selections (based on CERN paper)
dF_total = apply_selection_threshold(dF_total, 'mu_plus_PT', 800)
dF_total = apply_selection_threshold(dF_total, 'mu_minus_PT', 800)
dF_total = apply_selection_threshold(dF_total, 'K_PT', 250)
dF_total = apply_selection_threshold(dF_total, 'Pi_PT', 250)
dF_filtered_out = pd.concat([dF_filtered_out, apply_selection_threshold(dF_unfiltered, 'mu_plus_PT', 800, opposite=True)],ignore_index=True).drop_duplicates()
dF_filtered_out = pd.concat([dF_filtered_out, apply_selection_threshold(dF_unfiltered, 'mu_minus_PT', 800, opposite=True)],ignore_index=True).drop_duplicates()
dF_filtered_out = pd.concat([dF_filtered_out, apply_selection_threshold(dF_unfiltered, 'K_PT', 250, opposite=True)],ignore_index=True).drop_duplicates()
dF_filtered_out = pd.concat([dF_filtered_out, apply_selection_threshold(dF_unfiltered, 'Pi_PT', 250, opposite=True)],ignore_index=True).drop_duplicates()
print('Total data selection criteria done')

# Observable fitting parameters
ll_dG_dctl.errordef = Minuit.LIKELIHOOD
ll_dG_dctk.errordef = Minuit.LIKELIHOOD
ll_dG_dphi.errordef = Minuit.LIKELIHOOD
decimal_places = 3
starting_point = [-0.1, 0.0]
fls_l, fl_errs_l, fls_k, fl_errs_k, fls_p, fl_errs_p = [], [], [], [], [], []
afbs, afb_errs = [], []
ats, at_errs = [], []
aims, aim_errs = [], []

legobj_backgr_k, legobj_backgr_l, legobj_backgr_p = Background.background(dF_total, total=True)

max_degree = 4
acceptance_k, acceptance_l, acceptance_p, x_interval_for_leg = Acceptance.acceptance()
legendre_k = L.legfit(x_interval_for_leg, acceptance_k, max_degree)
legendre_l = L.legfit(x_interval_for_leg, acceptance_l, max_degree)
legendre_p = L.legfit(x_interval_for_leg, acceptance_p, max_degree)


# SEPARATING DATA INTO q^2 BINS
q_ranges = [[0.01,0.98],[1.1,2.5],[2.5,4.0],[4.0,6.0],[6.0,8.0],[15.0,17.0],[17.0,19.0],[11.0,12.5],[1.0,6.0],[15.0,17.9]]
q_ranges_paper = [[0.01,2.0],[2.0,4.0],[4.0,8.5],[10.0,13.0],[14.5,16.0],[16.0,23.0]]

for q_range in q_ranges:
    print('q range: ', q_range)
    mask = (dF_total['q2'] > q_range[0]) & (dF_total['q2'] < q_range[1])
    dF = dF_total[mask]

    mean, sigma, sig_ratio, noi_ratio, sig_events, noi_events = Background.background(dF)

    significance = 3.0
    lower_bound = mean - significance*sigma
    upper_bound = mean + significance*sigma
    sig_mask = (dF['B0_MM'] > lower_bound) & (dF['B0_MM'] < upper_bound)
    noise_mask = (dF['B0_MM'] > upper_bound)
    signal = dF[sig_mask]
    noise = dF[noise_mask]

    # Bins of histograms
    y_sigk, bin_borders_sigk = np.histogram(signal['costhetak'], bins='auto')
    x_sigk = bin_borders_sigk[:-1] + np.diff(bin_borders_sigk) / 2
    y_sigl, bin_borders_sigl = np.histogram(signal['costhetal'], bins='auto')
    x_sigl = bin_borders_sigl[:-1] + np.diff(bin_borders_sigl) / 2
    y_sigp, bin_borders_sigp = np.histogram(signal['phi'], bins='auto')
    x_sigp = (bin_borders_sigp[:-1] + np.diff(bin_borders_sigp) / 2) / np.pi
    y_back_k = L.legval(x_sigk, legobj_backgr_k)
    y_back_l = L.legval(x_sigl, legobj_backgr_l)
    y_back_p = L.legval(x_sigp, legobj_backgr_p)

    # Background Normalisation Constants
    B_k = noi_events/sum(y_back_k)
    B_l = noi_events/sum(y_back_l)
    B_p = noi_events/sum(y_back_p)
    print('Background Normalisation Coefficients: ')
    print(B_k)
    print(B_l)
    print(B_p)

    y_sigk -= (B_k*y_back_k).astype('int')
    y_sigl -= (B_l*y_back_l).astype('int')
    y_sigp -= (B_p*y_back_p).astype('int')

    # acceptance normalisation constants
    A_k = sp.integrate.simps(y_sigk,x_sigk) / sp.integrate.simps(y_sigk*L.legval(x_sigk, legendre_k),x_sigk)
    A_l = sp.integrate.simps(y_sigl,x_sigl) / sp.integrate.simps(y_sigl*L.legval(x_sigl, legendre_l),x_sigl)
    A_p = sp.integrate.simps(y_sigp,x_sigp) / sp.integrate.simps(y_sigp*L.legval(x_sigp, legendre_p),x_sigp)
    print('Acceptance Normalisation Coefficients: ')
    print(A_k)
    print(A_l)
    print(A_p)

    y_sigk_acc = A_k*y_sigk*L.legval(x_sigk, legendre_k)
    y_sigl_acc = A_l*y_sigl*L.legval(x_sigl, legendre_l)
    y_sigp_acc = A_p*y_sigp*L.legval(x_sigp, legendre_p)

    p_k = L.legfit(x_sigk, y_sigk_acc, max_degree)
    p_l = L.legfit(x_sigl, y_sigl_acc, max_degree)
    p_p = L.legfit(x_sigp, y_sigp_acc, max_degree)
    y_k = L.legval(x_interval_for_leg, p_k)
    y_l = L.legval(x_interval_for_leg, p_l)
    y_p = L.legval(x_interval_for_leg, p_p)

    """
    #plt.bar(x_sigk, y_sigk, width=(x_sigk[-1]-x_sigk[0])/(len(x_sigk)-1), label='signal', zorder=2)
    plt.bar(x_sigk, y_sigk_acc,  width=(x_sigk[-1]-x_sigk[0])/(len(x_sigk)-1), label='signal_acc', zorder=1)
    #plt.bar(x_noik, y_noik,  width=(x_sigk[-1]-x_sigk[0])/(len(x_sigk)-1), label='noise', zorder=3)
    plt.plot(x_interval_for_leg, y_k, color='red', label='fit to data after bkgr and acc', zorder=4)
    plt.legend()
    plt.title('Total data set: cos(theta_k)')
    plt.show()

    #plt.bar(x_sigl, y_sigl, width=(x_sigl[-1]-x_sigl[0])/(len(x_sigl)-1), label='signal', zorder=1)
    plt.bar(x_sigl, y_sigl_acc, width=(x_sigl[-1]-x_sigl[0])/(len(x_sigl)-1), label='signal_acc', zorder=2)
    #plt.bar(x_noil, y_noil, width=(x_sigl[-1]-x_sigk[0])/(len(x_sigl)-1), label='noise', zorder=3)
    plt.plot(x_interval_for_leg, y_l, color='red', label='fit to data after bkgr and acc', zorder=4)
    plt.legend()
    plt.title('Total data set: cos(theta_l)')
    plt.show()

    #plt.bar(x_sigp, y_sigp, width=(x_sigp[-1]-x_sigp[0])/(len(x_sigp)-1), label='signal', zorder=1)
    plt.bar(x_sigp, y_sigp_acc, width=(x_sigp[-1]-x_sigp[0])/(len(x_sigp)-1), label='signal_acc', zorder=2)
    #plt.bar(x_noip, y_noip, width=(x_sigp[-1]-x_sigp[0])/(len(x_sigp)-1), label='noise', zorder=3)
    plt.plot(x_interval_for_leg, y_p, color='red', label='fit to data after bkgr and acc', zorder=4)
    plt.legend()
    plt.title('Total data set: cos(theta_l)')
    plt.show()
    """

    # FITTING FOR THE OBSERVABLES PER q^2 BIN
    p0 = [0.0]
    fitParams_k, fitCovariances_k = curve_fit(dgamma_dcosthetak, x_sigk, y_sigk, p0)
    p0 = [-0.1, 0.0]
    fitParams_l, fitCovariances_l = curve_fit(dgamma_dcosthetal, x_sigl, y_sigl, p0)
    p0 = [-0.1, 0.0, 0.1]
    fitParams_p, fitCovariances_p = curve_fit(dgamma_dphi, x_sigp, y_sigp, p0)
    """
    l = Minuit(ll_dG_dctl, fl=starting_point[0], afb=starting_point[1], data=dF)
    k = Minuit(ll_dG_dctk, fl=starting_point[0], data=dF)
    p = Minuit(ll_dG_dphi, fl=starting_point[0], at=starting_point[0], aim=starting_point[0], data=dF)

    l.fixed['data'] = True  # fixing the bin number as we don't want to optimize it
    l.limits=((-1.0, 1.0), (-1.0, 1.0), None)
    l.migrad()
    l.hesse()

    k.fixed['data'] = True  # fixing the bin number as we don't want to optimize it
    k.limits = ((-1.0, 1.0), None)
    k.migrad()
    k.hesse()

    p.fixed['data'] = True  # fixing the bin number as we don't want to optimize it
    p.limits = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), None)
    p.migrad()
    p.hesse()
    """

    # Append all the values
    fls_k.append(fitParams_k[0])
    fl_errs_k.append(fitCovariances_k[0][0])

    fls_l.append(fitParams_l[0])
    afbs.append(fitParams_l[1])
    fl_errs_l.append(fitCovariances_l[0][0])
    afb_errs.append(fitCovariances_l[1][1])

    fls_p.append(fitParams_p[0])
    ats.append(fitParams_p[1])
    aims.append(fitParams_p[2])
    fl_errs_p.append(fitCovariances_p[0][0])
    at_errs.append(fitCovariances_p[1][1])
    aim_errs.append(fitCovariances_p[2][2])


print(len(q_ranges))
print(len(fls_k))
print(len(fl_errs_k))

### PLOTTING OBSERVABLES
plotting_bool = True
plt.rcParams.update({'errorbar.capsize': 2, 'text.usetex': True})

if plotting_bool == True:
    plt.figure()
    plt.title("Values of $F_{L}$ for all 3 distributions (l, k \& phi) per $q^2$ bin")
    #plt.errorbar(range(0,len(q_ranges)),fls_p, yerr=fl_errs_p, marker='x', markersize=6, linestyle='none')
    plt.errorbar(range(0,len(q_ranges)),fls_l, yerr=fl_errs_l, marker='x', markersize=6, linestyle='none')
    plt.errorbar(range(0,len(q_ranges)),fls_k, yerr=fl_errs_k, marker='x', markersize=6, linestyle='none')
    plt.legend(['$F_{Lphi}$','$F_{Ll}$','$F_{Lk}$'])
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Values of $A_{FB}$ per $q^2$ bin")
    plt.errorbar(range(0,len(q_ranges)),afbs, yerr=afb_errs, marker='x', markersize=6, linestyle='none')
    plt.legend(['$A_{FB}$'])
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Values of $A_{T}$ per $q^2$ bin")
    plt.errorbar(range(0,len(q_ranges)),ats, yerr=at_errs, marker='x', markersize=6, linestyle='none')
    plt.legend(['$A_{T}$'])
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Values of $A_{im}$ per $q^2$ bin")
    plt.errorbar(range(0,len(q_ranges)),aims, yerr=aim_errs, marker='x', markersize=6, linestyle='none')
    plt.legend(['$A_{im}$'])
    plt.grid()
    plt.show()

