
import numpy as np 
import os
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import pandas as pd

def Gaussian(x, mean, sigma, A): 
    '''Gaussian Function'''
    
    Gauss = A * np.e**(-(((x-mean)/sigma))**2/2)
    return Gauss

def criteria_for_interest(sigma_sig, mean_sig, mean_background, Nsigma = 10): 
    '''checks whether the mean values of the background and signal are 3 sigmas apart'''
    
    if Nsigma * sigma_sig < np.absolute(mean_sig - mean_background):
        return True
    else: 
        return False

def plot_and_fit(dF_signal, dF_background, background, parameter, bins = 70):
    
    
    def gaussian_fit(dF, label, color):
        '''fitting a Gaussian over the said parameter given data'''
        
        hist = plt.hist(dF[str(parameter)], bins = bins, label = label, color = color, alpha = 0.5, density = True) 
        
        bin_heights = hist [0]
        bin_edges = hist[1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        #obtaining initial guess
        initial_A = max(bin_heights)
        initial_mean = bin_centers[int(len(bin_centers)/2)]
        initial_sigma = np.std(bin_centers)
        initial_guess = [initial_mean, initial_sigma, initial_A]
        
        #curve fit to Gaussian
        
        try:
            popt, pcov = curve_fit(Gaussian, bin_centers, bin_heights, p0 = initial_guess)
        except:
            
            return None, None
        
        plt.plot(bin_edges, Gaussian(bin_edges, *popt), label = label, color = color, alpha = 0.7)
        
        return popt, pcov 
    
    #plotting graphs
    plt.title('Comparison of signal and %s with parameter %s ' % (background, parameter))
    plt.xlabel('%s' % (parameter))
    plt.ylabel('Events')
    lowerLim = min(min(dF_background[str(parameter)]), min(dF_signal[str(parameter)]))
    upperLim = max(max(dF_background[str(parameter)]), max(dF_signal[str(parameter)]))
    
    
    popt_sig, pcov_sig = gaussian_fit(dF_signal, 'signal', color = 'blue')
    popt_background, pcov_background = gaussian_fit(dF_background, str(background), color = 'red')
    plt.xlim([lowerLim, upperLim])
    plt.savefig('Parameters_of_interests_graphs/signal_and_%s_%s.png' % (background, parameter))
    plt.legend()
    plt.show()
    try:
        mean_fitted_background, sigma_fitted_background, A_fitted_background = popt_background
        mean_fitted_sig, sigma_fitted_sig, A_fitted_sig = popt_sig 
        return sigma_fitted_sig, mean_fitted_sig, mean_fitted_background
    except: 
        return None, None, None

'''Set your paths here'''

datafolder_path = 'C:/Users/leehi/OneDrive/Documents/Imperial_tings/Third_year/Team_based_problem_solving/group5-ic-teach-kstmumu/year3-problem-solving/year3-problem-solving/csv/'
signal = "sig.csv"
dF_signal = pd.read_csv(os.path.join(datafolder_path, signal))

background_list = ['jpsi.csv', 'jpsi_mu_k_swap.csv', 'jpsi_mu_pi_swap.csv', 'k_pi_swap.csv', 
                   'phimumu.csv', 'pKmumu_piTok_kTop.csv', 'pKmumu_piTop.csv','psi2S.csv']
parameter_list = [column for column in dF_signal.columns if 'Prob' not in column] 
#parameter_list = ['Kstar_MM', 'B0_MM']  ##to be commented!!!

#creates folder to store graphs
if not os.path.exists('Parameters_of_interests_graphs'):
    os.makedirs('Parameters_of_interests_graphs')

'''main loop'''
ofInterestDict = {}

for background in background_list: 
    
    dF_background = pd.read_csv(os.path.join(datafolder_path, background))
    
    for parameter in parameter_list: 
        
        print('current progress:', background, parameter)
        sigma_sig, mean_sig, mean_background = plot_and_fit(dF_signal, dF_background, background, parameter)
        try:
            ofInterest = criteria_for_interest(sigma_sig, mean_sig, mean_background)
        except: 
            ofInterest = 'To Check'
        ofInterestDict[str(background) + str(parameter)] = ofInterest