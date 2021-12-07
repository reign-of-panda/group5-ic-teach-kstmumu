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

def acceptance(folder_path, file_name):
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

    dF_acc['accept_kaon'] = dF_acc["K_MC15TuneV1_ProbNNk"] * (1 - dF_acc["K_MC15TuneV1_ProbNNp"])
    dF_acc['accept_pion'] = dF_acc["Pi_MC15TuneV1_ProbNNpi"] * (1 - dF_acc["Pi_MC15TuneV1_ProbNNk"]) * (1 - dF_acc["Pi_MC15TuneV1_ProbNNp"])
    dF_acc['accept_muon'] = dF_acc[['mu_plus_MC15TuneV1_ProbNNmu', 'mu_plus_MC15TuneV1_ProbNNmu']].max(axis=1)

    dF_acc_unfiltered = dF_acc

    # Apply selection criteria to acceptance data
    dF_acc = apply_selection_threshold(dF_acc_unfiltered, 'accept_kaon', 0.05)
    dF_acc = apply_selection_threshold(dF_acc, 'accept_pion', 0.1)
    dF_acc = apply_selection_threshold(dF_acc, 'accept_muon', 0.2)
    dF_acc_filtered_out = apply_selection_threshold(dF_acc_unfiltered, 'accept_kaon', 0.05, opposite=True)
    dF_acc_filtered_out = pd.concat([dF_acc_filtered_out, apply_selection_threshold(dF_acc_unfiltered, 'accept_pion', 0.1, opposite=True)], ignore_index=True).drop_duplicates()
    dF_acc_filtered_out = pd.concat([dF_acc_filtered_out, apply_selection_threshold(dF_acc_unfiltered, 'accept_muon', 0.2, opposite=True)], ignore_index=True).drop_duplicates()

    # Transverse momenta selections (based on CERN paper)
    dF_acc = apply_selection_threshold(dF_acc, 'mu_plus_PT', 800)
    dF_acc = apply_selection_threshold(dF_acc, 'mu_minus_PT', 800)
    dF_acc = apply_selection_threshold(dF_acc, 'K_PT', 250)
    dF_acc = apply_selection_threshold(dF_acc, 'Pi_PT', 250)
    dF_acc_filtered_out = pd.concat([dF_acc_filtered_out, apply_selection_threshold(dF_acc_unfiltered, 'mu_plus_PT', 800, opposite=True)], ignore_index=True).drop_duplicates()
    dF_acc_filtered_out = pd.concat([dF_acc_filtered_out, apply_selection_threshold(dF_acc_unfiltered, 'mu_minus_PT', 800, opposite=True)], ignore_index=True).drop_duplicates()
    dF_acc_filtered_out = pd.concat([dF_acc_filtered_out, apply_selection_threshold(dF_acc_unfiltered, 'K_PT', 250, opposite=True)], ignore_index=True).drop_duplicates()
    dF_acc_filtered_out = pd.concat([dF_acc_filtered_out, apply_selection_threshold(dF_acc_unfiltered, 'Pi_PT', 250, opposite=True)], ignore_index=True).drop_duplicates()
    #print(len(dF_acc),len(dF_acc_unfiltered),len(dF_acc_filtered_out))
    print('Acceptance selection criteria done')

    acceptance_unsel = dF_acc_unfiltered
    acceptance_sel = dF_acc

    # Histogram bins
    bin_heights_acc_uk, bin_borders_acc_uk = np.histogram(acceptance_unsel['costhetak'], bins='auto')
    bin_centers_acc_uk = bin_borders_acc_uk[:-1] + np.diff(bin_borders_acc_uk) / 2
    bin_heights_acc_sk, bin_borders_acc_sk = np.histogram(acceptance_sel['costhetak'], bins='auto')
    bin_centers_acc_sk = bin_borders_acc_sk[:-1] + np.diff(bin_borders_acc_sk) / 2
    bin_heights_acc_ul, bin_borders_acc_ul = np.histogram(acceptance_unsel['costhetal'], bins='auto')
    bin_centers_acc_ul = bin_borders_acc_ul[:-1] + np.diff(bin_borders_acc_ul) / 2
    bin_heights_acc_sl, bin_borders_acc_sl = np.histogram(acceptance_sel['costhetal'], bins='auto')
    bin_centers_acc_sl = bin_borders_acc_sl[:-1] + np.diff(bin_borders_acc_sl) / 2

    max_degree = 5
    x_interval_for_leg = np.linspace(-1.0, 1.0, 100)

    print('Started fitting histograms')
    p_uk = L.legfit(bin_centers_acc_uk, bin_heights_acc_uk, max_degree)
    y_uk = L.legval(x_interval_for_leg, p_uk)
    p_ul = L.legfit(bin_centers_acc_ul, bin_heights_acc_ul, max_degree)
    y_ul = L.legval(x_interval_for_leg, p_ul)

    p_sk = L.legfit(bin_centers_acc_sk, bin_heights_acc_sk, max_degree)
    y_sk = L.legval(x_interval_for_leg, p_sk)
    p_sl = L.legfit(bin_centers_acc_sl, bin_heights_acc_sl, max_degree)
    y_sl = L.legval(x_interval_for_leg, p_sl)


    # We want to multiply the selected data with the ratio of unselected/selected
    # so that we retrieve a 'unselected distribution' after selection

    #must be of the form: unselected/selected
    redist_k = L.legval(x_interval_for_leg,L.legfit(x_interval_for_leg, y_uk/y_sk, max_degree))
    redist_l = L.legval(x_interval_for_leg,L.legfit(x_interval_for_leg, y_ul/y_sl, max_degree))

    normalisation_k = sp.integrate.simps(y_sk, x_interval_for_leg)/sp.integrate.simps(y_sk*redist_k, x_interval_for_leg)
    normalisation_l = sp.integrate.simps(y_sl, x_interval_for_leg)/sp.integrate.simps(y_sl*redist_l, x_interval_for_leg)

    acceptance_k = redist_k*normalisation_k
    acceptance_l = redist_l*normalisation_l

    print(normalisation_k)
    print(normalisation_l)

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

    return acceptance_k, acceptance_l, x_interval_for_leg