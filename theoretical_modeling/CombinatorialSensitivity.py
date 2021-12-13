
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from operator import truediv


folder_path = "/Users/raymondvanes/Downloads"
file_name = "/total_dataset.csv"
file_path = folder_path + file_name
dF = pd.read_csv(file_path)

K_ProbNNp = dF["K_MC15TuneV1_ProbNNp"]
K_probNNk = dF["K_MC15TuneV1_ProbNNk"]
Pi_ProbNNp = dF["Pi_MC15TuneV1_ProbNNp"]
Pi_probNNk = dF["Pi_MC15TuneV1_ProbNNk"]
Pi_ProbNNpi = dF["Pi_MC15TuneV1_ProbNNpi"]

dF['accept_kaon'] = K_probNNk * (1 - K_ProbNNp)
dF['accept_pion'] = Pi_ProbNNpi * (1 - Pi_probNNk) * (1 - Pi_ProbNNp)
dF['accept_muon'] = dF[['mu_plus_MC15TuneV1_ProbNNmu', 'mu_plus_MC15TuneV1_ProbNNmu']].max(axis=1)


def gauss_exp(x, a, mean, sigma, A, b):
    return a * np.exp(-((x - mean) ** 2 / (2 * sigma ** 2))) + A * np.exp(b * (x - 5170))
def gauss(x, a, mean, sigma):
    return a * np.exp(-((x - mean) ** 2 / (2 * sigma ** 2)))
def exp_tail(x, A, b):
    return A * np.exp(b * (x - 5170))

# Number of different thressholds per variable
steps = 100
# List of the relevant variables, some have been removed due to poor results already.
variable_list = [['accept_kaon',1], ['accept_pion',1], ['accept_muon',1],
                ['mu_plus_MC15TuneV1_ProbNNk',0],
                ['mu_plus_MC15TuneV1_ProbNNpi',0], ['mu_plus_MC15TuneV1_ProbNNmu',1],
                ['mu_plus_MC15TuneV1_ProbNNe',0], ['mu_plus_MC15TuneV1_ProbNNp',0],

                ['mu_plus_P',1], ['mu_plus_PT',1], ['mu_plus_ETA',1],
                ['mu_plus_PE',1],

                ['K_MC15TuneV1_ProbNNk',1], ['K_MC15TuneV1_ProbNNpi',0],
                ['K_MC15TuneV1_ProbNNe',0], ['K_MC15TuneV1_ProbNNp',0],

                ['K_P',0], ['K_PT',0], ['K_ETA',0], ['K_PE',0],

                ['Pi_MC15TuneV1_ProbNNk',0], ['Pi_MC15TuneV1_ProbNNpi',1],
                ['Pi_MC15TuneV1_ProbNNe',0], ['Pi_MC15TuneV1_ProbNNp',0],

                ['Pi_P',0], ['Pi_PT',0], ['Pi_ETA',0], ['Pi_PE',0],

                ['B0_ENDVERTEX_CHI2',0], ['B0_FDCHI2_OWNPV',0],
                ['Kstar_MM',0], ['Kstar_ENDVERTEX_CHI2',0], ['Kstar_FDCHI2_OWNPV',0],
                ['J_psi_MM',1], ['J_psi_ENDVERTEX_CHI2',0], ['J_psi_FDCHI2_OWNPV',0],
                ['B0_IPCHI2_OWNPV',0], ['B0_DIRA_OWNPV',1], ['B0_FD_OWNPV',1]]

# Will eventually store the threshold per variable
threshold_dict = {}
plot_gaussians = False

# The idea of this script is to loop through each variable and compare how much the signal events decrease vs the noise
# events for thresholds ranging from the minimum to the maximum value of the variable.
# The way the signal ratio decrease is calculated:
# is by applying the threshold to the data => figuring out the mean and standard deviation of the
# gaussian(+exponential) of the B0_MM graph => using this range figure out the signal ratio i.e. how many events fall
# in this range before applying the threshold vs after applying the threshold.
# The noise ratio is calculated using the parameters of the gaussian+exponential fit of the B0_MM graph after
# applying the threshold (so exactly the same fit as was used for the signal ratio calculation). Then with those
# parameters we calculate the area under the purely gaussian bit of that curve and the area under the purely
# exponential bit and compare the 2 to get a noise ratio.
# Then we select the threshold which maximises the signal ratio while minimizing the noise ratio through scoring
# the threshold based on the metric: sig_ratio ^ n / noi_ratio
# Where n is included to address the overcutting produced when multiple thresholds are applied that all cut significant
# portions of the signal. Currently n=2.

for variable, bigger_than_bool in variable_list[25:-1]:

    print(variable)
    print(f'Min = {dF[variable].min()}, Max =  {dF[variable].max()}')

    # If we want to set a threshold s.t. the variable has to be larger them bigger_than_bool := 1
    # If we want the threshold s.t. the variable is lower than the threshold then bigger_than_bool :=0
    if bigger_than_bool == 1:
        threshold_list = np.linspace(dF[variable].min(), dF[variable].max(), steps)
    elif bigger_than_bool == 0:
        threshold_list = np.flip(np.linspace(dF[variable].min(),dF[variable].max(), steps))
    else:
        threshold_list = np.linspace(dF[variable].min(), dF[variable].max(), steps)

    # As you can see not the whole thresshold list is used. That is because the thresholds range from the minimum
    # to the maximum value of the variable. But as the threshold comes close to the maximum value the number of events
    # goes to zero.
    threshold_list = threshold_list[0:-int(0.2*len(threshold_list))]

    # initialized as ones cause later on to select the best threshold the metric will be sig_ratio^n/noi_ratio
    # which causes problems if noi_ratio = 0. Also the ^n factor is included to ensure that overcutting will be
    # less likely to occur.
    sig_ratios = np.zeros(len(threshold_list))
    noi_ratios = np.ones(len(threshold_list))
    i = 0

    for threshold in threshold_list:
        i+=1 # important for indexing results do not comment out
        #if (i % 20) == 0:
        #    print(i)

        if bigger_than_bool == 1:
            mask = (dF[variable] >= threshold)
        elif bigger_than_bool == 0:
            mask = (dF[variable] < threshold)
        else:
            mask = (dF[variable] >= threshold)

        dF_filter = dF[mask]

        # fitting gaussian+exponential to the filtered data.
        if plot_gaussians == True:
            bin_heights, bin_borders, what = plt.hist(dF_filter['B0_MM'], range=[5170, 5600], bins='auto', label='histogram') # for plotting purposes => uncomment
        else:
            bin_heights, bin_borders = np.histogram(dF_filter['B0_MM'], range=[5170, 5600], bins='auto')
        bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
        popt, pcov = curve_fit(gauss, bin_centers, bin_heights, p0=[5e3, 5.28e3, 20])  # , 5e2, -1e4
        popt1, pcov1 = curve_fit(gauss_exp, bin_centers, bin_heights, p0=[5e3, 5.28e3, 20, 3e2, -1e-4])

        mean = popt1[1]
        sigma = popt1[2]
        significance = 3.0
        lower_bound = mean - significance * sigma
        upper_bound = mean + significance * sigma

        #print(f'noise integrated: {quad(exp_tail, lower_bound, upper_bound, args=tuple(popt1[3:]))[0]}')
        #print(f'everything integrated: {quad(gauss_exp, lower_bound, upper_bound, args=tuple(popt1))}')


        if plot_gaussians == True:
            x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1],100)
            plt.plot(x_interval_for_fit, gauss(x_interval_for_fit, *popt1[0:3]), label='fit_gauss')
            plt.plot(x_interval_for_fit, gauss_exp(x_interval_for_fit, *popt1), label='fit_gaussexp')
            plt.plot(x_interval_for_fit, exp_tail(x_interval_for_fit, *popt1[3:]), label='guesswork')
            plt.legend()
            plt.show()

        df_Filt_Sig = dF_filter[(dF_filter['B0_MM'] > lower_bound) & (dF_filter['B0_MM'] < upper_bound)]
        df_Sig = dF[(dF['B0_MM'] > lower_bound) & (dF['B0_MM'] < upper_bound)]
        if len(df_Sig.index) == 0:
            break
        sig_ratio = len(df_Filt_Sig.index)/len(df_Sig.index)
        sig_ratios[i-1] = sig_ratio

        noi_ratio = quad(exp_tail, lower_bound, upper_bound, args=tuple(popt1[3:]))[0] / \
                    quad(gauss_exp, lower_bound, upper_bound, args=tuple(popt1))[0]
        noi_ratios[i-1] = noi_ratio

    plt.figure()
    plt.title(f'Sensitivity plot for variable: {variable}')
    plt.plot(threshold_list,sig_ratios, label='signal ratio')
    plt.plot(threshold_list,noi_ratios/max(noi_ratios), label='noise ratio')
    plt.grid()
    plt.legend()

    # calculating best threshold score, and corresponding signal and noise ratios
    best_result = max(list(map(truediv, sig_ratios**2, noi_ratios)))
    best_index = np.argmax(list(map(truediv, sig_ratios**2, noi_ratios)))
    #print(best_index)
    best_threshold = threshold_list[best_index]
    best_sig_rat = sig_ratios[best_index]
    best_noi_rat = noi_ratios[best_index]
    print(f'Best result = {best_result}, best threshold = {best_threshold}, '
          f'best signal ratio = {best_sig_rat}, best noise ratio = {best_noi_rat}')
    threshold_dict[variable] = [best_result, best_threshold]


print(threshold_dict)
plt.show()




