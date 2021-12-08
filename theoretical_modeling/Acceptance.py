# -*- coding: utf-8 -*-
# Made by NathanvEs - 9/11/2021

import numpy as np
import pandas as pd
import scipy as sp
import sys
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.special import legendre
from numpy.polynomial import legendre as L
from iminuit import Minuit
import peaking_functions


def acceptance(folder_path, file_name):
    # folder_path = "/Users/raymondvanes/Downloads/splitcsv_acceptance"
    # file_name = "/acceptance_mc-3.csv"
    file_path = folder_path + file_name
    dF_acc = pd.read_csv(file_path)
    print('Reading acceptance file done')

    def apply_selection_threshold(dataF, column, threshold, opposite=False):
        mask = (dataF[column] >= threshold)
        if opposite == True:
            dataF = dataF[~mask]
        else:
            dataF = dataF[mask]
        return dataF
    
    def apply_selection_threshold_for_lambda(dataF, column, low, high, opposite=False):
        """
        Generic function for applying a selection criteria
        """
        mask = (dataF[column] < low ) | (dataF[column] > high )
        if opposite == True:
            dataF = dataF[~mask]
        else:
            dataF = dataF[mask]
        return dataF

    dF_acc['accept_kaon'] = dF_acc["K_MC15TuneV1_ProbNNk"] * (1 - dF_acc["K_MC15TuneV1_ProbNNp"])
    dF_acc['accept_pion'] = dF_acc["Pi_MC15TuneV1_ProbNNpi"] * (1 - dF_acc["Pi_MC15TuneV1_ProbNNk"]) * (1 - dF_acc["Pi_MC15TuneV1_ProbNNp"])
    dF_acc['accept_muon'] = dF_acc[['mu_plus_MC15TuneV1_ProbNNmu', 'mu_plus_MC15TuneV1_ProbNNmu']].max(axis=1)

    dF_acc_unfiltered = dF_acc

    # Apply selection criteria to acceptance data
    dF_acc = apply_selection_threshold(dF_acc_unfiltered, 'accept_kaon', 0.1)
    dF_acc = apply_selection_threshold(dF_acc, 'accept_pion', 0.1)
    # dF_acc = apply_selection_threshold(dF_acc, 'accept_muon', 0.2)
    dF_acc_filtered_out = apply_selection_threshold(dF_acc_unfiltered, 'accept_kaon', 0.1, opposite=True)
    dF_acc_filtered_out = pd.concat([dF_acc_filtered_out, apply_selection_threshold(dF_acc_unfiltered, 'accept_pion', 0.1, opposite=True)], ignore_index=True).drop_duplicates()
    # dF_acc_filtered_out = pd.concat([dF_acc_filtered_out, apply_selection_threshold(dF_acc_unfiltered, 'accept_muon', 0.2, opposite=True)], ignore_index=True).drop_duplicates()

    # Transverse momenta selections (based on CERN paper)
    dF_acc = apply_selection_threshold(dF_acc, 'mu_plus_PT', 3330)
    dF_acc = apply_selection_threshold(dF_acc, 'mu_minus_PT', 1000)
    dF_acc = apply_selection_threshold(dF_acc, 'K_PT', 1000)
    # dF_acc = apply_selection_threshold(dF_acc, 'Pi_PT', 250)
    dF_acc_filtered_out = pd.concat([dF_acc_filtered_out, apply_selection_threshold(dF_acc_unfiltered, 'mu_plus_PT', 3330, opposite=True)], ignore_index=True).drop_duplicates()
    dF_acc_filtered_out = pd.concat([dF_acc_filtered_out, apply_selection_threshold(dF_acc_unfiltered, 'mu_minus_PT', 1000, opposite=True)], ignore_index=True).drop_duplicates()
    dF_acc_filtered_out = pd.concat([dF_acc_filtered_out, apply_selection_threshold(dF_acc_unfiltered, 'K_PT', 1000, opposite=True)], ignore_index=True).drop_duplicates()
    # dF_acc_filtered_out = pd.concat([dF_acc_filtered_out, apply_selection_threshold(dF_acc_unfiltered, 'Pi_PT', 250, opposite=True)], ignore_index=True).drop_duplicates()
    #print(len(dF_acc),len(dF_acc_unfiltered),len(dF_acc_filtered_out))
    
    # Chi squared
    for chi2_param in ['mu_plus_IPCHI2_OWNPV', 'mu_minus_IPCHI2_OWNPV', 'K_IPCHI2_OWNPV', 'Pi_IPCHI2_OWNPV']:
        dF_acc = apply_selection_threshold(dF_acc, chi2_param, 16)
        dF_acc_filtered_out = pd.concat([dF_acc_filtered_out,
                                          apply_selection_threshold(dF_acc_unfiltered, chi2_param, 16,
                                                                         opposite=True)],
                                         ignore_index=True).drop_duplicates()
    
        dF_acc = apply_selection_threshold(dF_acc, 'B0_IPCHI2_OWNPV', 8.07, opposite=True)
        dF_acc_filtered_out = pd.concat([dF_acc_filtered_out,
                                          apply_selection_threshold(dF_acc_unfiltered, 'B0_IPCHI2_OWNPV', 8.07,
                                                                         opposite=False)],
                                         ignore_index=True).drop_duplicates()
    
    # Peaking Background
    #phimumu 
    pmm_bg = "/phimumu.csv"
    pmm_bg_path = folder_path + pmm_bg
    phimumu_data = pd.read_csv(pmm_bg_path)
    
    Phi_M = peaking_functions.phimumu(dF_acc)
    Phi_M_out = peaking_functions.phimumu(dF_acc_filtered_out)
    Phi_M_bg = peaking_functions.phimumu(phimumu_data)
    mu, sigma = sp.stats.norm.fit(Phi_M_bg)
    # mask = (Phi_M >= mu+sigma)
    dF_acc['Phi_M'] = Phi_M
    dF_acc_filtered_out['Phi_M'] = Phi_M_out
    dF_acc = apply_selection_threshold(dF_acc, 'Phi_M', mu+sigma)
    dF_acc_filtered_out = apply_selection_threshold(dF_acc_filtered_out, 'Phi_M', mu+sigma, opposite=True)
    # dF_acc = dF_acc[mask]
    # dF_acc_filtered_out = dF_acc_filtered_out[~mask]
    
    
    #pKmumu_piTop
    lambda1 = "\pKmumu_piTop.csv"
    lambda1_path = folder_path + lambda1
    lambda1_data = pd.read_csv(lambda1_path)
    lambda1_M = peaking_functions.pKmumu_piTop(dF_acc)
    lambda1_M_out = peaking_functions.pKmumu_piTop(dF_acc_filtered_out)
    
    # lambda1_M = peaking_functions.pKmumu_piTop(dF_acc)
    lambda1_M_bg = peaking_functions.pKmumu_piTop(lambda1_data)
    mu, sigma = sp.stats.norm.fit(lambda1_M_bg)
    low = mu - sigma
    high = mu + sigma
    
    dF_acc['lambda1_M'] = lambda1_M
    dF_acc_filtered_out['lambda1_M'] = lambda1_M_out
    # mask = (lambda1_M < low ) | (lambda1_M > high)
    dF_acc = apply_selection_threshold_for_lambda(dF_acc, 'lambda1_M', low, high)
    dF_acc_filtered_out = apply_selection_threshold_for_lambda(dF_acc_filtered_out, 'lambda1_M', low, high, opposite = True)
  
    
    #pKmumu_piTok_kTop
    
    lambda2 = "\pKmumu_piTok_kTop.csv"
    lambda2_path = folder_path + lambda2
    lambda2_data = pd.read_csv(lambda2_path)
    lambda2_M = peaking_functions.pKmumu_piTok_kTop(dF_acc)
    lambda2_M_out = peaking_functions.pKmumu_piTok_kTop(dF_acc_filtered_out)
    
    # lambda2_M = peaking_functions.pKmumu_piTok_kTop(dF_acc)
    lambda2_M_bg = peaking_functions.pKmumu_piTok_kTop(lambda2_data)
    mu, sigma = sp.stats.norm.fit(lambda2_M_bg)
    low = mu - sigma
    high = mu + sigma
    
    dF_acc['lambda2_M'] = lambda2_M
    dF_acc_filtered_out['lambda2_M'] = lambda2_M_out
    # mask = (lambda2_M < low ) | (lambda2_M > high )
    dF_acc = apply_selection_threshold_for_lambda(dF_acc, 'lambda2_M', low, high)
    dF_acc_filtered_out = apply_selection_threshold_for_lambda(dF_acc_filtered_out, 'lambda2_M', low, high, opposite = True)
    
    # From sensitivity analysis
    
    # params = ['J_psi_MM', 'K_ETA', 'K_P', 'J_psi_ENDVERTEX_CHI2']
    # cuts = [1725, 2.4, 20000, 0.52]
    # length = len(cuts)
    # for i in range(length):
    #     dF_acc = apply_selection_threshold(dF_acc, params[i], cuts[i])
    #     dF_acc_filtered_out = pd.concat([dF_acc_filtered_out,
    #                                       apply_selection_threshold(dF_acc_filtered_out, params[i], cuts[i],
    #                                                                      opposite=True)],
    #                                      ignore_index=True).drop_duplicates()
    print('Acceptance selection criteria done')

    acceptance_unsel = dF_acc_unfiltered
    acceptance_sel = dF_acc

    # Histogram bins
    # unselected u, selected s, costhetak k, ...
    bin_heights_acc_uk, bin_borders_acc_uk = np.histogram(acceptance_unsel['costhetak'], bins='auto')
    bin_centers_acc_uk = bin_borders_acc_uk[:-1] + np.diff(bin_borders_acc_uk) / 2
    bin_heights_acc_sk, bin_borders_acc_sk = np.histogram(acceptance_sel['costhetak'], bins='auto')
    bin_centers_acc_sk = bin_borders_acc_sk[:-1] + np.diff(bin_borders_acc_sk) / 2
    bin_heights_acc_ul, bin_borders_acc_ul = np.histogram(acceptance_unsel['costhetal'], bins='auto')
    bin_centers_acc_ul = bin_borders_acc_ul[:-1] + np.diff(bin_borders_acc_ul) / 2
    bin_heights_acc_sl, bin_borders_acc_sl = np.histogram(acceptance_sel['costhetal'], bins='auto')
    bin_centers_acc_sl = bin_borders_acc_sl[:-1] + np.diff(bin_borders_acc_sl) / 2
    bin_heights_acc_up, bin_borders_acc_up = np.histogram(acceptance_unsel['phi'], bins='auto')
    bin_centers_acc_up = (bin_borders_acc_up[:-1] + np.diff(bin_borders_acc_up) / 2) / np.pi
    bin_heights_acc_sp, bin_borders_acc_sp = np.histogram(acceptance_sel['phi'], bins='auto')
    bin_centers_acc_sp = (bin_borders_acc_sp[:-1] + np.diff(bin_borders_acc_sp) / 2) / np.pi

    max_degree = 4
    x_interval_for_leg = np.linspace(-1.0, 1.0, 100)

    print('Started fitting histograms')
    p_uk = L.legfit(bin_centers_acc_uk, bin_heights_acc_uk, max_degree)
    y_uk = L.legval(x_interval_for_leg, p_uk)
    p_ul = L.legfit(bin_centers_acc_ul, bin_heights_acc_ul, max_degree)
    y_ul = L.legval(x_interval_for_leg, p_ul)
    p_up = L.legfit(bin_centers_acc_up, bin_heights_acc_up, max_degree)
    y_up = L.legval(x_interval_for_leg, p_up)

    p_sk = L.legfit(bin_centers_acc_sk, bin_heights_acc_sk, max_degree)
    y_sk = L.legval(x_interval_for_leg, p_sk)
    p_sl = L.legfit(bin_centers_acc_sl, bin_heights_acc_sl, max_degree)
    y_sl = L.legval(x_interval_for_leg, p_sl)
    p_sp = L.legfit(bin_centers_acc_sp, bin_heights_acc_sp, max_degree)
    y_sp = L.legval(x_interval_for_leg, p_sp)


    # We want to multiply the selected data with the ratio of unselected/selected
    # so that we retrieve a 'unselected distribution' after selection

    #must be of the form: unselected/selected
    redist_k = L.legval(x_interval_for_leg,L.legfit(x_interval_for_leg, y_uk/y_sk, max_degree))
    redist_l = L.legval(x_interval_for_leg,L.legfit(x_interval_for_leg, y_ul/y_sl, max_degree))
    redist_p = L.legval(x_interval_for_leg,L.legfit(x_interval_for_leg, y_up/y_sp, max_degree))

    normalisation_k = sp.integrate.simps(y_sk, x_interval_for_leg)/sp.integrate.simps(y_sk*redist_k, x_interval_for_leg)
    normalisation_l = sp.integrate.simps(y_sl, x_interval_for_leg)/sp.integrate.simps(y_sl*redist_l, x_interval_for_leg)
    normalisation_p = sp.integrate.simps(y_sp, x_interval_for_leg)/sp.integrate.simps(y_sp*redist_p, x_interval_for_leg)

    acceptance_k = redist_k*normalisation_k
    acceptance_l = redist_l*normalisation_l
    acceptance_p = redist_p*normalisation_p

    print(normalisation_k)
    print(normalisation_l)
    print(normalisation_p)

    """
    plt.hist(acceptance_unsel['costhetak'], bins='auto', label='unselected', zorder=1)
    plt.hist(acceptance_sel['costhetak'], bins='auto', label='selected', zorder=2)
    plt.plot(x_interval_for_leg, y_uk, zorder=3)
    plt.plot(x_interval_for_leg, y_sk, zorder=4)
    plt.plot(x_interval_for_leg, y_sk*acceptance_k, zorder=5)
    plt.legend()
    plt.title('Acceptance data set: cos(theta_k)')
    plt.show()

    plt.hist(acceptance_unsel['costhetal'], bins='auto', label='unselected')
    plt.hist(acceptance_sel['costhetal'], bins='auto', label='selected')
    plt.plot(x_interval_for_leg, y_ul, zorder=3)
    plt.plot(x_interval_for_leg, y_sl, zorder=4)
    plt.plot(x_interval_for_leg, y_sl*acceptance_l, zorder=5)
    plt.legend()
    plt.title('Acceptance data set: cos(theta_l)')
    plt.show()

    plt.hist(acceptance_unsel['phi'], bins='auto', label='unselected')
    plt.hist(acceptance_sel['phi'], bins='auto', label='selected')
    plt.plot(x_interval_for_leg, y_up, zorder=3)
    plt.plot(x_interval_for_leg, y_sp, zorder=4)
    plt.plot(x_interval_for_leg, y_sp*acceptance_p, zorder=5)
    plt.legend()
    plt.title('Acceptance data set: cos(theta_l)')
    plt.show()
    """

    return redist_k, redist_l, redist_p, x_interval_for_leg