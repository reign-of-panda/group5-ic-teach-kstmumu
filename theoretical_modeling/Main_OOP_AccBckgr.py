# -*- coding: utf-8 -*-
# Made by NathanvEs - 15/10/2021

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.stats import norm 
from numpy.polynomial import legendre as L
import Acceptance
import Background
import Predictions

import peaking_functions


class LHCb:
    def __init__(self):
        """
        Loads in data
        """
        # Change this path to whatever it is on your personal computer
        self.folder_path = "D:\OneDrive - Imperial College London\Imperial College London\Module Content\Year 3\Problem Solving\year3-problem-solving\year3-problem-solving\csv"
        file_name = "/total_dataset.csv"
        file_path = self.folder_path + file_name
        self.dF = pd.read_csv(file_path)
        self.acceptance_file = "/acceptance_mc.csv"
        
        
        print('Reading total dataset done')

        self.dF_unfiltered = self.dF

        self.max_degree = 4

    def Mass(self, PE, P):
        """
        Returns the mass in units of MeV/c^2
        """
        return (PE ** 2 - P ** 2) ** 0.5

    def apply_selection_threshold(self, dataF, column, threshold, opposite=False):
        """
        Generic function for applying a selection criteria
        """
        mask = (dataF[column] >= threshold)
        if opposite == True:
            dataF = dataF[~mask]
        else:
            dataF = dataF[mask]
        return dataF
    
    def apply_selection_threshold_for_lambda(self, dataF, column, low, high, opposite=False):
        """
        Generic function for applying a selection criteria
        """
        mask = (dataF[column] < low ) | (dataF[column] > high )
        if opposite == True:
            dataF = dataF[~mask]
        else:
            dataF = dataF[mask]
        return dataF
    

    def peaking(self): 
            """
            Taking away peaking backgrounds via reconstruction (includes phimumu, 
                                                                and the 2 lambda).
            """
            #phimumu 
            pmm_bg = "/phimumu.csv"
            pmm_bg_path = self.folder_path + pmm_bg
            phimumu_data = pd.read_csv(pmm_bg_path)
            
            Phi_M = peaking_functions.phimumu(self.dF)
            Phi_M_out = peaking_functions.phimumu(self.dF_filtered_out)
            Phi_M_bg = peaking_functions.phimumu(phimumu_data)
            mu, sigma = sp.stats.norm.fit(Phi_M_bg)
            # mask = (Phi_M >= mu+sigma)
            # mask2 = (Phi_M >= mu+sigma)
            self.dF['Phi_M'] = Phi_M
            self.dF = self.apply_selection_threshold(self.dF, 'Phi_M', mu+sigma)
            self.dF_filtered_out['Phi_M'] = Phi_M_out
            self.dF_filtered_out = self.apply_selection_threshold(self.dF_filtered_out, 'Phi_M', mu+sigma, opposite = True)
            
            
            # self.dF = self.dF[mask]
            # self.dF_filtered_out = self.dF_filtered_out[~mask]
            
            
            #pKmumu_piTop
            lambda1 = "\pKmumu_piTop.csv"
            lambda1_path = self.folder_path + lambda1
            lambda1_data = pd.read_csv(lambda1_path)
            lambda1_M = peaking_functions.pKmumu_piTop(self.dF)
            lambda1_M_out = peaking_functions.pKmumu_piTop(self.dF_filtered_out)
            
            # lambda1_M = peaking_functions.pKmumu_piTop(self.dF)
            lambda1_M_bg = peaking_functions.pKmumu_piTop(lambda1_data)
            mu, sigma = sp.stats.norm.fit(lambda1_M_bg)
            low = mu - sigma
            high = mu + sigma
            
            self.dF['lambda1_M'] = lambda1_M    
            self.dF_filtered_out['lambda1_M'] = lambda1_M_out 
            # mask = (lambda1_M < low ) | (lambda1_M > high )
            self.dF = self.apply_selection_threshold_for_lambda(self.dF, 'lambda1_M', low, high)
            self.dF_filtered_out = self.apply_selection_threshold_for_lambda(self.dF_filtered_out, 'lambda1_M', low, high, opposite = True)
      
            
            #pKmumu_piTok_kTop
            
            lambda2 = "\pKmumu_piTok_kTop.csv"
            lambda2_path = self.folder_path + lambda2
            lambda2_data = pd.read_csv(lambda2_path)
            lambda2_M = peaking_functions.pKmumu_piTok_kTop(self.dF)
            lambda2_M_out = peaking_functions.pKmumu_piTok_kTop(self.dF_filtered_out)
            
            # lambda2_M = peaking_functions.pKmumu_piTok_kTop(self.dF)
            lambda2_M_bg = peaking_functions.pKmumu_piTok_kTop(lambda2_data)
            mu, sigma = sp.stats.norm.fit(lambda2_M_bg)
            low = mu - sigma
            high = mu + sigma
            
            self.dF['lambda2_M'] = lambda2_M 
            self.dF_filtered_out['lambda2_M'] = lambda2_M_out
            # mask = (lambda2_M < low ) | (lambda2_M > high )
            self.dF = self.apply_selection_threshold_for_lambda(self.dF, 'lambda2_M', low, high)
            self.dF_filtered_out = self.apply_selection_threshold_for_lambda(self.dF_filtered_out, 'lambda2_M', low, high, opposite = True)


    def probability_assignment(self):
        """
        Assigning probability variables
        """
        mu_plus_ProbNNmu = self.dF['mu_plus_MC15TuneV1_ProbNNmu']
        mu_minus_ProbNNmu = self.dF['mu_minus_MC15TuneV1_ProbNNmu']
        mu_plus_ProbNNp = self.dF["mu_plus_MC15TuneV1_ProbNNp"]
        mu_plus_probNNk = self.dF["mu_plus_MC15TuneV1_ProbNNk"]
        mu_plus_ProbNNpi = self.dF["mu_plus_MC15TuneV1_ProbNNpi"]
        K_ProbNNp = self.dF["K_MC15TuneV1_ProbNNp"]
        K_probNNk = self.dF["K_MC15TuneV1_ProbNNk"]
        Pi_ProbNNp = self.dF["Pi_MC15TuneV1_ProbNNp"]
        Pi_probNNk = self.dF["Pi_MC15TuneV1_ProbNNk"]
        Pi_ProbNNpi = self.dF["Pi_MC15TuneV1_ProbNNpi"]

        """
        Some selection criteria
        P(kaon): ProbNNK · (1 − ProbNNp) > 0.05 
        P(pion): ProbNNπ · (1 − ProbNNK) · (1 − ProbNNp) > 0.1
        """

        self.dF['accept_kaon'] = K_probNNk * (1 - K_ProbNNp)
        self.dF['accept_pion'] = Pi_ProbNNpi * (1 - Pi_probNNk) * (1 - Pi_ProbNNp)
        self.dF['accept_muon'] = self.dF[['mu_plus_MC15TuneV1_ProbNNmu', 'mu_plus_MC15TuneV1_ProbNNmu']].max(axis=1)
        self.dF['dilepton_mass'] = self.Mass(self.dF['mu_minus_PE'], self.dF['mu_minus_P']) + self.Mass(self.dF['mu_plus_PE'], self.dF['mu_plus_P'])

    def probability_filter(self, order):
        """
        Filtering the data based on the probabilities
        """

        if order == 0:
            # self.dF_unfiltered = self.dF

            # Probability selections (based on CERN paper)
            self.dF = self.apply_selection_threshold(self.dF_unfiltered, 'accept_kaon', 0.1)
            self.dF = self.apply_selection_threshold(self.dF, 'accept_pion', 0.1)
            # self.dF = self.apply_selection_threshold(self.dF, 'accept_muon', 0.2)
            self.dF_filtered_out = self.apply_selection_threshold(self.dF_unfiltered, 'accept_kaon', 0.1,
                                                                  opposite=True)
            self.dF_filtered_out = pd.concat([self.dF_filtered_out,
                                              self.apply_selection_threshold(self.dF_unfiltered, 'accept_pion', 0.1,
                                                                             opposite=True)],
                                             ignore_index=True).drop_duplicates()
            # self.dF_filtered_out = pd.concat([self.dF_filtered_out,
            #                                   self.apply_selection_threshold(self.dF_unfiltered, 'accept_muon', 0.2,
            #                                                                  opposite=True)],
            #                                  ignore_index=True).drop_duplicates()
        else:
            self.dF = self.apply_selection_threshold(self.dF, 'accept_kaon', 0.1)
            self.dF = self.apply_selection_threshold(self.dF, 'accept_pion', 0.1)
            # self.dF = self.apply_selection_threshold(self.dF, 'accept_muon', 0.2)
            self.dF_filtered_out = pd.concat([self.dF_filtered_out,
                                              self.apply_selection_threshold(self.dF_unfiltered, 'accept_kaon', 0.1,
                                                                             opposite=True)],
                                             ignore_index=True).drop_duplicates()
            self.dF_filtered_out = pd.concat([self.dF_filtered_out,
                                              self.apply_selection_threshold(self.dF_unfiltered, 'accept_pion', 0.1,
                                                                             opposite=True)],
                                             ignore_index=True).drop_duplicates()
            # self.dF_filtered_out = pd.concat([self.dF_filtered_out,
            #                                   self.apply_selection_threshold(self.dF_unfiltered, 'accept_muon', 0.2,
            #                                                                  opposite=True)],
            #                                  ignore_index=True).drop_duplicates()

            print(len(self.dF), len(self.dF_unfiltered), len(self.dF_filtered_out))

    def trasv_mom_filter(self, order):

        if order == 0:
            self.dF_unfiltered = self.dF

            # Transverse momenta selections (based on CERN paper)
            self.dF = self.apply_selection_threshold(self.dF_unfiltered, 'mu_plus_PT', 3330)
            self.dF = self.apply_selection_threshold(self.dF, 'mu_minus_PT', 1000)
            self.dF = self.apply_selection_threshold(self.dF, 'K_PT', 1000)
            # self.dF = self.apply_selection_threshold(self.dF, 'Pi_PT', 250)
            self.dF_filtered_out = self.apply_selection_threshold(self.dF_unfiltered, 'mu_plus_PT', 800, opposite=True)
            self.dF_filtered_out = pd.concat([self.dF_filtered_out, self.apply_selection_threshold(self.dF_unfiltered, 'mu_minus_PT', 800,opposite=True)], ignore_index=True).drop_duplicates()
            self.dF_filtered_out = pd.concat([self.dF_filtered_out, self.apply_selection_threshold(self.dF_unfiltered, 'K_PT', 250, opposite=True)], ignore_index=True).drop_duplicates()
            # self.dF_filtered_out = pd.concat([self.dF_filtered_out, self.apply_selection_threshold(self.dF_unfiltered, 'Pi_PT', 250, opposite=True)], ignore_index=True).drop_duplicates()

        else:
            # Transverse momenta selections (based on CERN paper)
            self.dF = self.apply_selection_threshold(self.dF, 'mu_plus_PT', 3330)
            self.dF = self.apply_selection_threshold(self.dF, 'mu_minus_PT', 1000)
            self.dF = self.apply_selection_threshold(self.dF, 'K_PT', 1000)
            # self.dF = self.apply_selection_threshold(self.dF, 'Pi_PT', 250)
            self.dF_filtered_out = pd.concat([self.dF_filtered_out, self.apply_selection_threshold(self.dF_unfiltered, 'mu_plus_PT', 3330, opposite=True)], ignore_index=True).drop_duplicates()
            self.dF_filtered_out = pd.concat([self.dF_filtered_out, self.apply_selection_threshold(self.dF_unfiltered, 'mu_minus_PT', 1000, opposite=True)], ignore_index=True).drop_duplicates()
            self.dF_filtered_out = pd.concat([self.dF_filtered_out, self.apply_selection_threshold(self.dF_unfiltered, 'K_PT', 1000, opposite=True)], ignore_index=True).drop_duplicates()
            # self.dF_filtered_out = pd.concat([self.dF_filtered_out, self.apply_selection_threshold(self.dF_unfiltered, 'Pi_PT', 250, opposite=True)], ignore_index=True).drop_duplicates()

            print(len(self.dF), len(self.dF_unfiltered), len(self.dF_filtered_out))

    def chi_sq_filter(self):
        for chi2_param in ['mu_plus_IPCHI2_OWNPV', 'mu_minus_IPCHI2_OWNPV', 'K_IPCHI2_OWNPV', 'Pi_IPCHI2_OWNPV']:
            self.dF = self.apply_selection_threshold(self.dF, chi2_param, 16)
            self.dF_filtered_out = pd.concat([self.dF_filtered_out,
                                              self.apply_selection_threshold(self.dF_unfiltered, chi2_param, 16,
                                                                             opposite=True)],
                                             ignore_index=True).drop_duplicates()

        self.dF = self.apply_selection_threshold(self.dF, 'B0_IPCHI2_OWNPV', 8.07, opposite=True)
        self.dF_filtered_out = pd.concat([self.dF_filtered_out,
                                          self.apply_selection_threshold(self.dF_unfiltered, 'B0_IPCHI2_OWNPV', 8.07,
                                                                         opposite=False)],
                                         ignore_index=True).drop_duplicates()
        
    def from_sensitivity_analysis(self):
        params = ['J_psi_MM', 'K_ETA', 'K_P', 'J_psi_ENDVERTEX_CHI2']
        cuts = [1725, 2.4, 20000, 0.52]
        length = len(cuts)
        
        for i in range(length):
            self.dF = self.apply_selection_threshold(self.dF, params[i], cuts[i])
            self.dF_filtered_out = pd.concat([self.dF_filtered_out,
                                              self.apply_selection_threshold(self.dF_unfiltered, params[i], cuts[i],
                                                                             opposite=True)],
                                             ignore_index=True).drop_duplicates()
        

    def intermediary_plotting(self):
        # Some plotting
        n1, bin1, patches1 = plt.hist(self.dF_unfiltered['B0_MM'], range=[5170, 5600], bins=300, zorder=1,
                                      histtype="step")
        n2, bin2, patches2 = plt.hist(self.dF['B0_MM'], range=[5170, 5600], bins=300, zorder=3, histtype="step")
        n3, bin3, patches3 = plt.hist(self.dF_filtered_out['B0_MM'], range=[5170, 5600], bins=300, zorder=2,
                                      histtype="step")
        plt.title('Invariant Mass of $B_0$ with & without background')
        plt.xlabel('MM($B_0$)(MeV/$c^2$)')
        plt.ylabel('Number of Candidates')
        plt.legend(['unfiltered', 'filtered', 'filtered out'])

        # Fitting to the invariant mass plot
        def gauss_exp(x, a, b, c, e, f):
            return a * np.exp(-((x - b) ** 2) / 2 / c ** 2) + np.exp(e * x + f)

        def gauss(x, a, b, c):
            return a * np.exp(-((x - b) ** 2) / 2 / c ** 2)

        def exp_tail(x, b, c):
            return np.exp(b * x + c)

        bin1_alt = []
        for i in range(len(n1)):
            bin1_alt.append((bin1[i] + bin1[i + 1]) / 2)

        bin2_alt = []
        for i in range(len(n1)):
            bin2_alt.append((bin2[i] + bin2[i + 1]) / 2)

        bin3_alt = []
        for i in range(len(n1)):
            bin3_alt.append((bin3[i] + bin3[i + 1]) / 2)

        guess2 = [7000, 5270, 20]
        params2, cov2 = curve_fit(gauss, bin2_alt, n2, p0=guess2)

        guess3 = [3000, 5270, 20] + [-0.0007, 9.5]
        params3, cov3 = curve_fit(gauss_exp, bin3_alt, n3, p0=guess3)

        guess1 = params2.tolist() + params3.tolist()[-2:]

        params1, cov1 = curve_fit(gauss_exp, bin1_alt, n1, p0=guess1)

        x_fit = np.linspace(5170, 5600, 500)
        fit1 = gauss_exp(x_fit, *params1)
        fit2 = gauss(x_fit, *params2)
        fit3 = gauss_exp(x_fit, *params3)

        #plt.plot(x_fit, fit1, color = "blue", zorder=4)
        #plt.plot(x_fit, fit2, color = "red", zorder=5)
        #plt.plot(x_fit, fit3, color = "black", zorder=6)
        #plt.show()

    def totaldata_backgr_acc(self):
        self.legobj_backgr_k, self.legobj_backgr_l, self.legobj_backgr_p = Background.background(self.dF, total=True)
        
        self.acceptance_k, self.acceptance_l, self.acceptance_p, self.x_interval_for_leg = Acceptance.acceptance(self.folder_path,self.acceptance_file)
        self.legendre_k = L.legfit(self.x_interval_for_leg, self.acceptance_k, self.max_degree)
        self.legendre_l = L.legfit(self.x_interval_for_leg, self.acceptance_l, self.max_degree)
        self.legendre_p = L.legfit(self.x_interval_for_leg, self.acceptance_p, self.max_degree)

    def q_separate(self):
        """
        SEPARATING DATA INTO q^2 BINS
        """
        self.bins = []
        self.q_ranges = [[0.01, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], [15.0, 17.0], [17.0, 19.0],
                    [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]
        q_ranges_paper = [[0.01, 2.0], [2.0, 4.0], [4.0, 8.5], [10.0, 13.0], [14.5, 16.0], [16.0, 23.0]]

        for q_range in self.q_ranges:
            mask = (self.dF['q2'] > q_range[0]) & (self.dF['q2'] < q_range[1])
            bin = self.dF[mask]
            self.bins.append(bin)

    def backgr_acceptance_fit_observables(self):

        self.fls_l, self.fl_errs_l, self.fls_k, self.fl_errs_k, self.fls_p, self.fl_errs_p = [], [], [], [], [], []
        self.afbs, self.afb_errs = [], []
        self.ats, self.at_errs = [], []
        self.aims, self.aim_errs = [], []
        bin = 1
        for dF in self.bins:
            print('Bin ' + str(bin) + ' out of ' + str(len(self.q_ranges)))
            print('q^2 range: ' + str(self.q_ranges[bin-1]))
            bin += 1
            mean, sigma, sig_ratio, noi_ratio, sig_events, noi_events = Background.background(dF)

            def dgamma_dcosthetak(fl):
                ctk = dF['costhetak']
                acceptance = L.legval(ctk, self.acceptance_k)
                background = L.legval(ctk, self.legobj_backgr_k)
                normalised_background = noi_ratio * (background/sp.integrate.simps(ctk, background))
                function = acceptance * ((3 / 2) * fl * ctk ** 2 + (3 / 4) * (1 - fl) * (1 - ctk ** 2) - normalised_background)
                normalised_scalar_array = function/sp.integrate.simps(function, ctk)
                return - np.sum(np.log(normalised_scalar_array))

            def dgamma_dcosthetal(fl, afb):
                ctl = dF['costhetal']
                acceptance = L.legval(ctl, self.acceptance_l)
                background = L.legval(ctl, self.legobj_backgr_l)
                normalised_background = noi_ratio * (background/sp.integrate.simps(ctl, background))
                function = acceptance * ((3 / 4) * fl * (1 - ctl ** 2) + (3 / 8) * (1 - fl) * (1 + ctl ** 2) + afb * ctl - normalised_background)
                normalised_scalar_array = function/sp.integrate.simps(function, ctl)
                return - np.sum(np.log(normalised_scalar_array))

            def dgamma_dphi(fl, at, aim):
                phi = dF['phi']
                acceptance = L.legval(phi, self.acceptance_p)
                background = L.legval(phi, self.legobj_backgr_p)
                normalised_background = noi_ratio * (background/sp.integrate.simps(phi, background))
                function = acceptance * ((1/(2 * np.pi)) * (1+(1/2)*(1 - fl) * at * np.cos(2 * phi) + aim * np.sin(2 * phi)) - normalised_background)
                normalised_scalar_array = function/sp.integrate.simps(function, phi)
                return - np.sum(np.log(normalised_scalar_array))


            dgamma_dcosthetak.errordef = Minuit.LIKELIHOOD
            dgamma_dcosthetal.errordef = Minuit.LIKELIHOOD
            dgamma_dphi.errordef = Minuit.LIKELIHOOD
            starting_point = [0.1, 0.0]

            k = Minuit(dgamma_dcosthetak, fl=starting_point[0])
            l = Minuit(dgamma_dcosthetal, fl=starting_point[0], afb=starting_point[0])
            p = Minuit(dgamma_dphi, fl=starting_point[0], at=starting_point[0], aim=starting_point[1])

            k.limits = (-1.0, 1.0)
            k.migrad()
            k.hesse()

            l.limits = ((-1.0, 1.0), (-1.0, 1.0))
            l.migrad()
            l.hesse()

            p.limits = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
            p.migrad()
            p.hesse()

            # Append all the values
            self.fls_k.append(k.values[0])
            self.fl_errs_k.append(k.errors[0])

            self.fls_l.append(l.values[0])
            self.afbs.append(l.values[1])
            self.fl_errs_l.append(l.errors[0])
            self.afb_errs.append(l.errors[1])

            self.fls_p.append(p.values[0])
            self.ats.append(p.values[1])
            self.aims.append(p.values[2])
            self.fl_errs_p.append(p.errors[0])
            self.at_errs.append(p.errors[1])
            self.aim_errs.append(p.errors[2])

            """"# FITTING FOR THE OBSERVABLES PER q^2 BIN
            p0 = [0.0]
            fitParams_k, fitCovariances_k = curve_fit(dgamma_dcosthetak, x_sigk, y_sigk_acc, p0)
            p0 = [-0.0, 0.0]
            fitParams_l, fitCovariances_l = curve_fit(dgamma_dcosthetal, x_sigl, y_sigl_acc, p0)
            p0 = [-0.1, 0.0, 0.1]
            fitParams_p, fitCovariances_p = curve_fit(dgamma_dphi, x_sigp, y_sigp_acc, p0)

            # Append all the values
            self.fls_k.append(fitParams_k[0])
            self.fl_errs_k.append(fitCovariances_k[0][0])

            self.fls_l.append(fitParams_l[0])
            self.afbs.append(fitParams_l[1])
            self.fl_errs_l.append(fitCovariances_l[0][0])
            self.afb_errs.append(fitCovariances_l[1][1])

            self.fls_p.append(fitParams_p[0])
            self.ats.append(fitParams_p[1])
            self.aims.append(fitParams_p[2])
            self.fl_errs_p.append(fitCovariances_p[0][0])
            self.at_errs.append(fitCovariances_p[1][1])
            self.aim_errs.append(fitCovariances_p[2][2])"""

    def plot_observable(self):
        """
        PLOTTING OBSERVABLES
        """
        # Get predictions for observables (given by Mitesh)
        df_pred = Predictions.predictions()

        plotting_bool = True
        # plt.rcParams.update({'errorbar.capsize': 2, 'text.usetex': True})

        if plotting_bool == True:
            plt.figure()
            plt.title("Values of $F_{L}$ for all 3 distributions (l, k \& phi) per $q^2$ bin")
            plt.errorbar(range(0, len(self.bins)), self.fls_p, yerr=self.fl_errs_p, marker='x', markersize=6,
                          linestyle='none')
            plt.errorbar(range(0, len(self.bins)), self.fls_l, yerr=self.fl_errs_l, marker='x', markersize=6,
                          linestyle='none')
            plt.errorbar(range(0, len(self.bins)), self.fls_k, yerr=self.fl_errs_k, marker='x', markersize=6,
                          linestyle='none')
            plt.errorbar(range(0, len(self.bins)), df_pred['fl_Si'], yerr=df_pred['fl_Si_err'], marker='x', markersize=6,
                         linestyle='none')
            plt.errorbar(range(0, len(self.bins)), df_pred['fl_Pi'], yerr=df_pred['fl_Pi_err'], marker='x', markersize=6,
                         linestyle='none')
            plt.legend(['$F_{Lphi}$', '$F_{Ll}$', '$F_{Lk}$','$Fl_{Pred_S}$','$Fl_{Pred_P}$'], bbox_to_anchor = (1, 0.5))
            plt.grid()
            plt.show()

            plt.figure()
            plt.title("Values of $A_{FB}$ per $q^2$ bin")
            plt.errorbar(range(0, len(self.bins)), self.afbs, yerr=self.afb_errs, marker='x', markersize=6,
                         linestyle='none')
            plt.errorbar(range(0, len(self.bins)), df_pred['afb'], yerr=df_pred['afb_err'], marker='x', markersize=6,
                         linestyle='none')
            plt.legend(['$A_{FB}$','$A_{FB, Pred}$'])
            plt.grid()
            plt.show()

            plt.figure()
            plt.title("Values of $A_{T}$ per $q^2$ bin")
            plt.errorbar(range(0, len(self.bins)), self.ats, yerr=self.at_errs, marker='x', markersize=6,
                         linestyle='none')
            plt.errorbar(range(0, len(self.bins)), df_pred['at'], yerr=df_pred['at_err'], marker='x', markersize=6,
                         linestyle='none')
            plt.legend(['$A_{T}$','$A_{T, Pred}$'])
            plt.grid()
            plt.show()

            plt.figure()
            plt.title("Values of $A_{im}$ per $q^2$ bin")
            plt.errorbar(range(0, len(self.bins)), self.aims, yerr=self.aim_errs, marker='x', markersize=6,
                         linestyle='none')
            plt.errorbar(range(0, len(self.bins)), df_pred['aim'], yerr=df_pred['aim_err'], marker='x', markersize=6,
                         linestyle='none')
            plt.legend(['$A_{im}$','$A_{im, Pred}$'])
            plt.grid()
            plt.show()
            
    def save_plotting_data(self):
        """
        Saves some of the data so that we can make prettier plots
        """
        saving_directory = "D:/OneDrive - Imperial College London/Imperial College London/Module Content/Year 3/Problem Solving/plot_data"
        self.dF.to_csv(saving_directory + "/Filtered_data.csv") 
        self.dF_filtered_out.to_csv(saving_directory + "/Filtered_out_data.csv")
        
        # df_pred = Predictions.predictions()
        # df_pred.to_csv(saving_directory + "/Predictions.csv")
        
        # Observables
        fl_val = np.array([self.fls_p, self.fls_l, self.fls_k, self.fl_errs_p, self.fl_errs_l, self.fl_errs_k])
        afb_val_err = np.array([self.afbs, self.afb_errs])
        aims_val = np.array([self.aims, self.aim_errs])
        
        dF_fl = pd.DataFrame(data = fl_val.transpose(), columns=['fls_p', 'fls_l', 'fls_k', 'fl_errs_p', 'fl_errs_l', 'fl_errs_k'])
        dF_afb = pd.DataFrame(data = afb_val_err.transpose(), columns = ['afbs', 'afb_errs'])
        dF_aims = pd.DataFrame(data = aims_val.transpose(), columns = ['aims', 'aim_errs'])
        
        # Save this
        dF_fl.to_csv(saving_directory + "/fl_vals.csv")
        dF_afb.to_csv(saving_directory + "/afb_vals.csv")
        dF_aims.to_csv(saving_directory + "/aim_vals.csv")
        
    def run_analysis(self):
        """
        Runs the analysis

        set order = 0 to apply probability filter then transverse momentum filter
        set order = 1 to apply transverse momentum filter then probability filter

        """
        order = 0
        # Use the filtering functions first
        self.probability_assignment()
        if order == 0:
            self.probability_filter(order)
            self.trasv_mom_filter(1 - order)
        elif order == 1:
            self.trasv_mom_filter(1 - order)
            self.probability_filter(order)
            
        self.chi_sq_filter()
        self.peaking()
        # self.from_sensitivity_analysis()


        # Intermediary plot - comment out if not needed
        self.intermediary_plotting()
        self.totaldata_backgr_acc()

        # Separate q bins and Standard Model values
        self.q_separate()
        self.backgr_acceptance_fit_observables()
        self.plot_observable()
        
        self.save_plotting_data()


if __name__ == '__main__':
    CERN_Stuff = LHCb()
    CERN_Stuff.run_analysis()



