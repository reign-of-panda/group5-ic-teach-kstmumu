
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import pickle

def apply_selection_threshold(dataF, column_list, threshold_list, oppo_list):
    '''discards data based on thresholds'''
    
    for column, threshold, opposite in zip(column_list, threshold_list, oppo_list):
        mask = (dataF[column] >= threshold)
        if opposite == True:
            dataF = dataF[~mask]
        else:
            dataF = dataF[mask]
    return dataF

def generate_misidentify_prob_columns(dF): 
    '''generate misidentifying new columns as variables of sensitivity analysis'''
    
    #list of parameters to do with probability of misidentification
    prob_particles_list = [col for col in dF_signal_original.columns if 'Prob' in col]
    
    K_ProbNNp = dF['K_MC15TuneV1_ProbNNp']
    K_ProbNNk = dF['K_MC15TuneV1_ProbNNk']
    K_ProbNNpi = dF['K_MC15TuneV1_ProbNNpi']
    K_ProbNNmu = dF['K_MC15TuneV1_ProbNNmu']
    
    Pi_ProbNNp = dF['Pi_MC15TuneV1_ProbNNp']
    Pi_ProbNNk = dF['Pi_MC15TuneV1_ProbNNk']
    Pi_ProbNNpi = dF['Pi_MC15TuneV1_ProbNNpi']
    Pi_ProbNNmu = dF['Pi_MC15TuneV1_ProbNNmu']
    
    mu_plus_ProbNNp = dF['mu_plus_MC15TuneV1_ProbNNp']
    mu_plus_ProbNNk = dF['mu_plus_MC15TuneV1_ProbNNk']
    mu_plus_ProbNNpi = dF['mu_plus_MC15TuneV1_ProbNNpi']
    mu_plus_ProbNNmu = dF['mu_plus_MC15TuneV1_ProbNNmu']
    
    mu_minus_ProbNNp = dF['mu_minus_MC15TuneV1_ProbNNp']
    mu_minus_ProbNNk = dF['mu_minus_MC15TuneV1_ProbNNk']
    mu_minus_ProbNNpi = dF['mu_minus_MC15TuneV1_ProbNNpi']
    mu_minus_ProbNNmu = dF['mu_minus_MC15TuneV1_ProbNNmu']
     
    #according to formula from theory group 
    dF['accept_kaon'] = K_ProbNNk * (1 - K_ProbNNp) * (1 - K_ProbNNpi) * (1 - K_ProbNNmu)
    dF['accept_pion'] = Pi_ProbNNpi * (1 - Pi_ProbNNk) * (1 - Pi_ProbNNp) * (1 - Pi_ProbNNmu)
    dF['accept_mu_minus'] = mu_minus_ProbNNmu * (1 - mu_minus_ProbNNp) * (1 - mu_minus_ProbNNpi) * (1 - mu_minus_ProbNNmu)
    dF['accept_mu_plus'] = mu_plus_ProbNNk * (1 - mu_plus_ProbNNp) * (1 - mu_plus_ProbNNpi) * (1 - mu_plus_ProbNNmu)
    
    acceptance_columns = ['accept_kaon', 'accept_pion', 'accept_mu_minus', 'accept_mu_plus']
    
    return dF, acceptance_columns
    
def apply_threshold_filter(dataF, column, threshold, opposite=False):
    '''filter out data of a parameter column based on threshold'''
    
    N_initial = len(dataF[column])
    if opposite == True:
        data = [i for i in dataF[column] if i < threshold]
    else:
        data = [i for i in dataF[column] if i > threshold]
    N_final = len(data)
    
    return data, N_initial, N_final

def generate_thresholds(dF_background, parameter, divisions = 50): 
    '''divides parameter max and min into 50 steps'''
    
    maxValue = max(dF_background[parameter])
    minValue = min(dF_background[parameter])
    thresholds_list = np.linspace(minValue, maxValue, divisions)
    
    return thresholds_list

def plot_and_save_graphs(thresholds_list, R_list_signal, R_list_background, background, parameter):
    '''plots the line graphs, R against threshold values'''
    
    plt.clf()
    titleText = background + parameter
    plt.plot(thresholds_list, R_list_signal, color = 'blue', label = 'Signal') 
    plt.plot(thresholds_list, R_list_background, color = 'red', label = '%s' % background)
    #plt.scatter(thresholds_list, R_list_signal, color = 'blue', marker = '.') 
    #plt.scatter(thresholds_list, R_list_background, color = 'red', marker = '.')
    plt.legend()
    plt.xlabel('Threshold values')
    plt.ylabel('Proportion of Data Remaining (R)')
    plt.title('Proportion of data remaining due to different thresholds for %s and %s' % (background, parameter))
    plt.savefig('Sensitivity_Analysis_InPiority_Graphs/ %s.png' % titleText)
    plt.show()

    
'''set your paths here'''

datafolder_path = 'C:/Users/leehi/OneDrive/Documents/Imperial_tings/Third_year/Team_based_problem_solving/group5-ic-teach-kstmumu/year3-problem-solving/year3-problem-solving/csv/'
signal = 'sig.csv'
dF_signal_original = pd.read_csv(os.path.join(datafolder_path, signal))
dF_signal_original, acceptance_columns = generate_misidentify_prob_columns(dF_signal_original)

#customise the following 2 variables
background_list = ['jpsi.csv', 'jpsi_mu_k_swap.csv', 'jpsi_mu_pi_swap.csv', 'k_pi_swap.csv', 
                   'phimumu.csv', 'pKmumu_piTok_kTop.csv', 'pKmumu_piTop.csv','psi2S.csv']
#parameter_list = [column for column in dF_signal_original.columns if 'Prob' not in column]
acceptance_columns = ['accept_kaon', 'accept_pion', 'accept_mu_minus', 'accept_mu_plus']

#creates folder to store graphs
if not os.path.exists('Sensitivity_Analysis_InPiority_Graphs'):
    os.makedirs('Sensitivity_Analysis_InPiority_Graphs')
    
#creates folder to store thresholds and R data: 
if not os.path.exists('Sensitivity_Analysis_InPiority_Values'):
    os.makedirs('Sensitivity_Analysis_InPiority_Values')

'''main loop'''

parameter_list = ['J_psi_MM', 'J_psi_ENDVERTEX_CHI2', 'J_psi_ENDVERTEX_NDOF','J_psi_FDCHI2_OWNPV']

#parameters to apply thresholds on
parameters_to_apply_threshold = ['accept_kaon', 'accept_pion']
#list of thresholds to apply for parameters, every 8 
parameter_thresholds = np.array([[0,0,0,0,0,0,0,0],
                                 [0,0,0,0,0,0,0,0]])
#false ->  data below the threshold to be discard
opposite_list_orig = np.zeros(parameter_thresholds.shape, dtype = bool)


for background in background_list: 
    
    index = background_list.index(background)
    background_thresholds =  parameter_thresholds[:,index]
    opposite_list = opposite_list_orig[:, index]
    dF_signal = apply_selection_threshold(dF_signal_original, parameters_to_apply_threshold, background_thresholds, opposite_list)
    dF_background = pd.read_csv(os.path.join(datafolder_path, background))
    dF_background, acceptance_columns = generate_misidentify_prob_columns(dF_background)
    dF_background = apply_selection_threshold(dF_background, parameters_to_apply_threshold,  background_thresholds, opposite_list)
    
    for parameter in parameter_list:
        
        print('current progress:', background, parameter)
        
        thresholds_list = generate_thresholds(dF_background, parameter)
        R_list_signal = []
        R_list_background = []
        
        for threshold in thresholds_list: 
            
            _ , N_initial_background, N_final_background = apply_threshold_filter(dF_background, parameter, threshold)
            R_list_background.append(N_final_background/ N_initial_background )
            _ , N_initial_signal , N_final_signal = apply_threshold_filter(dF_signal, parameter, threshold)
            R_list_signal.append(N_final_signal/ N_initial_signal)
        
        plot_and_save_graphs(thresholds_list, R_list_signal, R_list_background, background, parameter)
        
        #saving values computed into pickle files
        pickleFileName = background + parameter
        with open('Sensitivity_Analysis_InPiority_Values/%s_threshold_values.pkl' % pickleFileName, 'wb') as f: 
            pickle.dump([thresholds_list, R_list_background, R_list_signal], f)
        
        
        

