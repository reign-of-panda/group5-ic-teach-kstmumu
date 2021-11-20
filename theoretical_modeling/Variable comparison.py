# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 18:33:57 2021

@author: user
"""

# this script compares different variables found in the signal and background files to determine approximate ranges of filters

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder_path = r'C:\Users\user\Documents\Imperial\Year 3\Comprehensives\TBPS\box_data_files' # change to whatever the path is on your system
background_names = ['jspi_background','psi2S_background','jpsi_mu_k_swap_background','jpsi_mu_pi_swap_background','k_pi_swap_background','phimumu_background','pKmumu_piTok_kTop_background','pKmumu_piTop_background']
particle_names = ['mu_plus', 'mu_minus', 'K', 'Pi']

def open_file(file_name): # opens the pickle file
    path = folder_path + file_name
    return pd.read_pickle(path)

def compare_hist(signal, backgrounds, variable): # compares the values of a selected variable for every background
    for i in range(len(backgrounds)):
        plt.figure(i)
        plt.hist(backgrounds[i][variable], bins = 300, label = 'background')
        plt.hist(signal[variable], bins = 300, label = 'signal')
        plt.title("signal vs. " + background_names[i])
        plt.xlabel(variable)
        plt.ylabel("Number of entries")
        plt.legend(loc = 'upper right')
      
def compare_with_dataset(signal, dataset, variable): # compares the signal values of a selected variable to that of the total dataset
    plt.figure(100)
    plt.hist(dataset[variable], bins = 300, label = 'dataset')
    plt.hist(signal[variable], bins = 300, label = 'signal')
    plt.title('Signal vs  dataset')
    plt.xlabel(variable)
    plt.ylabel("Number of entries")
    plt.legend(loc = 'upper right')

def Mass(dataframe, particle): # calculates the mass of a particle from energy and momentum data
    PE = dataframe[particle+"_PE"]
    P = dataframe[particle+"_P"]
    return (PE**2 - P**2)**0.5

def mass_comparison_hist(signal, background): # compares the mass of individual particles between signal a single background
    fig, axs = plt.subplots(2,2)
    fig.suptitle('mass distributions of signal vs. background')
    for i in range(len(particle_names)):
        signal_mass = Mass(signal, particle_names[i])
        background_mass = Mass(background, particle_names[i])
        print(signal_mass)
        if i < 2:  
            axs[0][i].hist(background_mass, bins = 300, label = 'background')
            axs[0][i].hist(signal_mass, bins = 300, label = 'signal')
            axs[0][i].set(xlabel = particle_names[i], ylabel = 'Number of entries')
            axs[0][i].legend(loc = 'upper right')    
        else:
            axs[1][i-2].hist(background_mass, bins = 300, label = 'background')
            axs[1][i-2].hist(signal_mass, bins = 300, label = 'signal')
            axs[1][i-2].set(xlabel = particle_names[i], ylabel = 'Number of entries')
            axs[1][i-2].legend(loc = 'upper right')
        
def all_mass_comparison_hist(signal, backgrounds): # compares the mass of individual particles between signal and all the backgrounds
    for j in range(len(background_names)):
        fig, axs = plt.subplots(2,2)
        fig.suptitle('mass distributions of signal vs. ' + background_names[j])
        for i in range(len(particle_names)):
            signal_mass = Mass(signal, particle_names[i])
            background_mass = Mass(backgrounds[j], particle_names[i])
            if i < 2: 
                axs[0][i].hist(background_mass, bins = 300, label = 'background')
                axs[0][i].hist(signal_mass, bins = 300, label = 'signal')
                axs[0][i].set(xlabel = particle_names[i], ylabel = 'Number of entries')
                axs[0][i].legend(loc = 'upper right')    
            else:
                axs[1][i-2].hist(background_mass, bins = 300, label = 'background')
                axs[1][i-2].hist(signal_mass, bins = 300, label = 'signal')
                axs[1][i-2].set(xlabel = particle_names[i], ylabel = 'Number of entries')
                axs[1][i-2].legend(loc = 'upper right')
                
def sum_mass_comparison_hist(signal, backgrounds): # compares the mass of ocmbinations of particles between signal and all the backgrounds
    for j in range(len(background_names)):
        
        fig, axs = plt.subplots(3)
        fig.suptitle('mass distributions of signal vs. ' + background_names[j])
        
        signal_lepton_mass = Mass(signal, particle_names[0]) + Mass(signal, particle_names[1])
        background_lepton_mass = Mass(backgrounds[j], particle_names[0]) + Mass(backgrounds[j], particle_names[1])

        signal_hadron_mass = Mass(signal, particle_names[2]) + Mass(signal, particle_names[3])        
        background_hadron_mass = Mass(backgrounds[j], particle_names[2]) + Mass(backgrounds[j], particle_names[3])
        
        signal_total_mass = signal_lepton_mass + signal_hadron_mass
        background_total_mass = background_lepton_mass + background_hadron_mass
                 
        axs[0].hist(background_lepton_mass, bins = 300, label = 'background')
        axs[0].hist(signal_lepton_mass, bins = 300, label = 'signal')
        axs[0].set(xlabel = "mu+ & mu-", ylabel = 'Number of entries')
        axs[0].legend(loc = 'upper right')    
        
        axs[1].hist(background_hadron_mass, bins = 300, label = 'background')
        axs[1].hist(signal_hadron_mass, bins = 300, label = 'signal')
        axs[1].set(xlabel = "K & Pi", ylabel = 'Number of entries')
        axs[1].legend(loc = 'upper right')
        
        axs[2].hist(background_total_mass, bins = 300, label = 'background')
        axs[2].hist(signal_total_mass, bins = 300, label = 'signal')
        axs[2].set(xlabel = "mu+ & mu- & K & Pi", ylabel = 'Number of entries')
        axs[2].legend(loc = 'upper right')
    
#%% loads in all the data files

dataset = open_file("/total_dataset.pkl")

signal = open_file("/sig.pkl")

jspi_background = open_file('/jpsi.pkl')

psi2S_background  = open_file('/psi2S.pkl')

jpsi_mu_k_swap_background = open_file('/jpsi_mu_k_swap.pkl')

jpsi_mu_pi_swap_background = open_file('/jpsi_mu_pi_swap.pkl')

k_pi_swap_background = open_file('/k_pi_swap.pkl')

phimumu_background = open_file('/phimumu.pkl')

pKmumu_piTok_kTop_background  = open_file('/pKmumu_piTok_kTop.pkl')

pKmumu_piTop_background = open_file('/pKmumu_piTop.pkl')

backgrounds = []
backgrounds.append(jspi_background)
backgrounds.append(psi2S_background)
backgrounds.append(jpsi_mu_k_swap_background)
backgrounds.append(jpsi_mu_pi_swap_background)
backgrounds.append(k_pi_swap_background)
backgrounds.append(phimumu_background)
backgrounds.append(pKmumu_piTok_kTop_background)
backgrounds.append(pKmumu_piTop_background)

#%% comparison of Kstar_MM data
compare_hist(signal, backgrounds, "Kstar_MM")
compare_with_dataset(signal, dataset, "Kstar_MM")

#%% comparison of B0_MM data
compare_hist(signal, backgrounds, "B0_MM")
compare_with_dataset(signal, dataset, "B0_MM")

#%% comparison of the mass of individual particles
all_mass_comparison_hist(signal, backgrounds)

#%% comparison of the mass of combinations of particles
sum_mass_comparison_hist(signal, backgrounds)

#%% Note for other users
'''
The above functions have been used to compare a few variables between signal and background datasets.

The same funcitons can be used to compare any other variable found in the data files.

This can be useful for determining where certain filters need to applied.

For example, running the "comparison of Kstar_MM data" cell shows that the signal has values of Kstar_MM concentrated in the region [800,1000], 
whilst some background decays have significant values outside this range, making it worthwile to apply a filter on Kstar_MM.
'''
