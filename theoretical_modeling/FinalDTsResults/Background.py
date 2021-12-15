# -*- coding: utf-8 -*-
# Made by NathanvEs - 23/11/2021

import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit
from numpy.polynomial import legendre as L
import matplotlib as plt


def gauss_exp(x, a, mean, sigma, A, b):
    return a * np.exp(-((x - mean) ** 2 / (2 * sigma ** 2))) + A * np.exp(b * (x - 5170))
def gauss(x, a, mean, sigma):
    return a * np.exp(-((x - mean) ** 2 / (2 * sigma ** 2)))
def exp_tail(x, A, b):
    return A * np.exp(b * (x - 5170))


def background(dF, total=False):
    #bin_heights, bin_borders, what = plt.hist(dF['B0_MM'], range=[5170, 5600], bins='auto', label='histogram') # for plotting purposes => uncomment
    bin_heights, bin_borders = np.histogram(dF['B0_MM'], range=[5170, 5600], bins='auto')
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    popt, pcov = curve_fit(gauss, bin_centers, bin_heights, p0=[5e3, 5.28e3, 20]) #, 5e2, -1e4
    popt1, pcov1 = curve_fit(gauss_exp, bin_centers, bin_heights, p0=[5e3, 5.28e3, 20, 3e2, -1e-4])
    print('Mean: ', popt1[1], ' +/- ', pcov1[1][1])
    print('Sigma: ', popt1[2], ' +/- ', pcov1[2][2])

    guess_exptail = [5e2, -1e-2]
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 100)
    #plt.plot(x_interval_for_fit, gauss(x_interval_for_fit, *popt), label='fit_gauss')
    #plt.plot(x_interval_for_fit, gauss_exp(x_interval_for_fit, *popt1), label='fit_gaussexp')
    #plt.plot(x_interval_for_fit, exp_tail(x_interval_for_fit, *guess_exptail), label='guesswork')
    #plt.legend()
    #plt.show()

    mean = popt1[1]
    sigma = popt1[2]
    significance = 3.0
    lower_bound = mean - significance*sigma
    upper_bound = mean + significance*sigma

    sig_ratio = quad(gauss, lower_bound, upper_bound, args=tuple(popt1[0:3]))[0] / quad(gauss_exp, lower_bound, upper_bound, args=tuple(popt1))[0]
    noi_ratio = quad(exp_tail, lower_bound, upper_bound, args=tuple(popt1[3:]))[0] / quad(gauss_exp, lower_bound, upper_bound, args=tuple(popt1))[0]

    sig_mask = (dF['B0_MM'] > lower_bound) & (dF['B0_MM'] < upper_bound)
    noise_mask = (dF['B0_MM'] > upper_bound)
    signal = dF[sig_mask]
    noise = dF[noise_mask]
    sig_events = signal.shape[0]*sig_ratio
    noi_events = signal.shape[0]*noi_ratio
    print(sig_ratio)
    print(noi_ratio)
    print(signal.shape[0])
    print(sig_events)
    print(noi_events)

    # Bins of histograms
    y_sigk, bin_borders_sigk = np.histogram(signal['costhetak'], bins='auto')
    x_sigk = bin_borders_sigk[:-1] + np.diff(bin_borders_sigk) / 2
    y_noik, bin_borders_noik = np.histogram(noise['costhetak'], bins=len(x_sigk))
    x_noik = bin_borders_noik[:-1] + np.diff(bin_borders_noik) / 2
    y_sigl, bin_borders_sigl = np.histogram(signal['costhetal'], bins='auto')
    x_sigl = bin_borders_sigl[:-1] + np.diff(bin_borders_sigl) / 2
    y_noil, bin_borders_noil = np.histogram(noise['costhetal'], bins=len(x_sigl))
    x_noil = bin_borders_noil[:-1] + np.diff(bin_borders_noil) / 2
    y_sigp, bin_borders_sigp = np.histogram(signal['phi'], bins='auto')
    x_sigp = (bin_borders_sigp[:-1] + np.diff(bin_borders_sigp) / 2) / np.pi
    y_noip, bin_borders_noip = np.histogram(noise['phi'], bins=len(x_sigp))
    x_noip = (bin_borders_noip[:-1] + np.diff(bin_borders_noip) / 2) / np.pi

    max_degree = 4
    backgr_k = L.legfit(x_noik, y_noik, max_degree)
    backgr_l = L.legfit(x_noil, y_noil, max_degree)
    backgr_p = L.legfit(x_noip, y_noip, max_degree)
    y_back_k = L.legval(x_noik, backgr_k)
    y_back_l = L.legval(x_noil, backgr_l)
    y_back_p = L.legval(x_noip, backgr_p)

    if total==True:
        return backgr_k, backgr_l, backgr_p
    else:
        return mean, sigma, sig_ratio, noi_ratio, sig_events, noi_events