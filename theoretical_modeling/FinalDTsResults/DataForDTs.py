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
        self.folder_path = "../data/csv"
        file_name = "/total_dataset.csv"
        file_path = self.folder_path + file_name
        self.dF = pd.read_csv(file_path)
        self.acceptance_file = "/acceptance_mc.csv"
        
        print('Reading total dataset done')
        
        self.readBg(self.folder_path) # for DTs
        print("Reading backgrounds done")

        self.dF_unfiltered = self.dF

        self.max_degree = 4

    def readBg(self, folder): # for DTs
        self.dF_jpsi = pd.read_csv(folder + "/jpsi.csv")
        self.dF_jpsiMuKSwap = pd.read_csv(folder + "/jpsi_mu_k_swap.csv")
        self.dF_jpsiMuPiSwap = pd.read_csv(folder + "/jpsi_mu_pi_swap.csv")
        self.dF_kPiSwap = pd.read_csv(folder + "/k_pi_swap.csv")
        self.dF_phimumu = pd.read_csv(folder + "/phimumu.csv")
        self.dF_pKmumuPiTok = pd.read_csv(folder + "/pKmumu_piTok_kTop.csv")
        self.dF_pKmumuPiTop = pd.read_csv(folder + "/pKmumu_piTop.csv")
        self.dF_psi2S = pd.read_csv(folder + "/psi2S.csv")
        
        self.peak_backgrounds = [self.dF_jpsi, self.dF_jpsiMuKSwap, self.dF_jpsiMuPiSwap, self.dF_kPiSwap, self.dF_phimumu, \
                              self.dF_pKmumuPiTok, self.dF_pKmumuPiTop, self.dF_psi2S]

    def Mass(self, PE, P):
        """
        Returns the mass in units of MeV/c^2
        """
        return (PE ** 2 - P ** 2) ** 0.5

    def apply_selection_threshold(self, dataF, column, threshold, opposite=False):
        """
        Generic function for applying a selection criteria
        """
        if dataF is self.dF: # for DTs
            self.copy_selection_threshold_forBG(column, threshold, opposite)
        
        mask = (dataF[column] >= threshold)
        if opposite == True:
            dataF = dataF[~mask]
        else:
            dataF = dataF[mask]
        return dataF
    
    def copy_selection_threshold_forBG(self, column, threshold, opposite): # for DTs
        for i, dataF in enumerate(self.peak_backgrounds):
            mask = (dataF[column] >= threshold)
            if opposite == True:
                self.peak_backgrounds[i] = dataF[~mask]
            else:
                self.peak_backgrounds[i] = dataF[mask]
    
    def apply_selection_threshold_for_lambda(self, dataF, column, low, high, opposite=False):
        """
        Generic function for applying a selection criteria
        """
        if dataF is self.dF: # for DTs
            self.copy_selection_threshold_lambda_forBG(column, low, high, opposite)
        
        mask = (dataF[column] < low ) | (dataF[column] > high )
        if opposite == True:
            dataF = dataF[~mask]
        else:
            dataF = dataF[mask]
        return dataF
    
    def copy_selection_threshold_lambda_forBG(self, column, low, high, opposite): # for DTs
        for i, dataF in enumerate(self.peak_backgrounds):
            mask = (dataF[column] < low ) | (dataF[column] > high )
            if opposite == True:
                self.peak_backgrounds[i] = dataF[~mask]
            else:
                self.peak_backgrounds[i] = dataF[mask]

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
            
            for i, df in enumerate(self.peak_backgrounds): # for DTs
                self.peak_backgrounds[i]['Phi_M'] = peaking_functions.phimumu(df)
            
            self.dF = self.apply_selection_threshold(self.dF, 'Phi_M', mu+0.5*sigma)
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
            low = mu - 0.4*sigma
            high = mu + 0.6*sigma
            
            self.dF['lambda1_M'] = lambda1_M    
            self.dF_filtered_out['lambda1_M'] = lambda1_M_out 
            
            for i, df in enumerate(self.peak_backgrounds): # for DTs
                self.peak_backgrounds[i]['lambda1_M'] = peaking_functions.pKmumu_piTop(df)
            
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
            low = mu - 0.5*sigma
            high = mu + 0.8*sigma
            
            self.dF['lambda2_M'] = lambda2_M 
            self.dF_filtered_out['lambda2_M'] = lambda2_M_out
            
            for i, df in enumerate(self.peak_backgrounds): # for DTs
                self.peak_backgrounds[i]['lambda2_M'] = peaking_functions.pKmumu_piTok_kTop(df)
            
            # mask = (lambda2_M < low ) | (lambda2_M > high )
            self.dF = self.apply_selection_threshold_for_lambda(self.dF, 'lambda2_M', low, high)
            self.dF_filtered_out = self.apply_selection_threshold_for_lambda(self.dF_filtered_out, 'lambda2_M', low, high, opposite = True)
            
            
            #K-muon swap 
            jpsi1 = "/jpsi_mu_k_swap.csv"
            jpsi1_path = self.folder_path + jpsi1
            jpsi1_data = pd.read_csv(jpsi1_path)
            
            jpsi1_MM = peaking_functions.jpsiKM2(self.dF)
            jpsi1_out = peaking_functions.jpsiKM2(self.dF_filtered_out)
            
            jpsi_BR= peaking_functions.jpsiKM2(jpsi1_data)
            mu,sigma = sp.stats.norm.fit(jpsi_BR)
            low = mu-2*sigma
            high = mu+2*sigma
            
            self.dF['jpsi1_MM'] = jpsi1_MM
            self.dF_filtered_out['jpsi1_MM'] = jpsi1_out
            
            for i, df in enumerate(self.peak_backgrounds): # for DTs
                self.peak_backgrounds[i]['jpsi1_MM'] = peaking_functions.jpsiKM2(df)
            
            self.dF = self.apply_selection_threshold_for_lambda(self.dF, 'jpsi1_MM', low, high)
            self.dF_filtered_out = self.apply_selection_threshold_for_lambda(self.dF_filtered_out, 'jpsi1_MM', low, high, opposite = True)

            #K-pion swap 
            jpsi2 = "/jpsi_mu_pi_swap.csv"
            jpsi2_path = self.folder_path + jpsi2
            jpsi2_data = pd.read_csv(jpsi2_path)
            
            jpsi2_MM = peaking_functions.jpsiPM(self.dF)
            jpsi2_out = peaking_functions.jpsiPM(self.dF_filtered_out)
            
            jpsi2_BR= peaking_functions.jpsiPM(jpsi2_data)
            mu,sigma = sp.stats.norm.fit(jpsi2_BR)
            low = mu-1.3*sigma
            high = mu+1.3*sigma
            
            self.dF['jpsi2_MM'] = jpsi2_MM
            self.dF_filtered_out['jpsi2_MM'] = jpsi2_out
            
            for i, df in enumerate(self.peak_backgrounds): # for DTs
                self.peak_backgrounds[i]['jpsi2_MM'] = peaking_functions.jpsiPM(df)
            
            self.dF = self.apply_selection_threshold_for_lambda(self.dF, 'jpsi2_MM', low, high)
            self.dF_filtered_out = self.apply_selection_threshold_for_lambda(self.dF_filtered_out, 'jpsi2_MM', low, high, opposite = True)

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
        self.dF['accept_muon'] = self.dF[['mu_plus_MC15TuneV1_ProbNNmu', 'mu_minus_MC15TuneV1_ProbNNmu']].max(axis=1)
        self.dF['dilepton_mass'] = self.Mass(self.dF['mu_minus_PE'], self.dF['mu_minus_P']) + self.Mass(self.dF['mu_plus_PE'], self.dF['mu_plus_P'])
        
        for i, df in enumerate(self.peak_backgrounds): # for DTs
            self.peak_backgrounds[i]['accept_kaon'] = df["K_MC15TuneV1_ProbNNk"] * (1 - df["K_MC15TuneV1_ProbNNp"])   
            self.peak_backgrounds[i]['accept_pion'] = df["Pi_MC15TuneV1_ProbNNpi"] * (1 - df["Pi_MC15TuneV1_ProbNNk"]) * (1 - df["Pi_MC15TuneV1_ProbNNp"])
            self.peak_backgrounds[i]['accept_muon'] = df[['mu_plus_MC15TuneV1_ProbNNmu', 'mu_minus_MC15TuneV1_ProbNNmu']].max(axis=1)
            self.peak_backgrounds[i]['dilepton_mass'] = self.Mass(df['mu_minus_PE'], df['mu_minus_P']) + self.Mass(df['mu_plus_PE'], df['mu_plus_P'])


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

        bin2_alt = []
        for i in range(len(n1)):
            bin2_alt.append((bin2[i] + bin2[i + 1]) / 2)

        guess2 = [7000, 5270, 20]
        params2, cov2 = curve_fit(gauss, bin2_alt, n2, p0=guess2)
        
        print("GAUSSIAN FITTING PARAMETERS: ", params2)
       
        
    def save_data(self):
        """
        Saves some of the data so that we can make prettier plots
        """
        saving_directory = "filtered_data"
        self.dF.to_csv(saving_directory + "/Filtered_data.csv") 
        self.dF_filtered_out.to_csv(saving_directory + "/Filtered_out_data.csv")
        
        self.save_peaking_Bg() # for DTs
        
    def save_peaking_Bg(self): # for DTs
        saving_directory = "peaking_bg_filtered"
        
        files = ["/jpsi.csv", "/jpsi_mu_k_swap.csv", "/jpsi_mu_pi_swap.csv", "/k_pi_swap.csv", "/phimumu.csv", "/pKmumu_piTok_kTop.csv", \
                 "/pKmumu_piTop.csv", "/psi2S.csv"]
        
        for i, df in enumerate(self.peak_backgrounds):
            df.to_csv(saving_directory + files[i])
    
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
        
        self.save_data()


if __name__ == '__main__':
    CERN_Stuff = LHCb()
    CERN_Stuff.run_analysis()


