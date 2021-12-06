
Guidelines: 
Divide the parameters related to your particle to five categories, and apply them in the following order: 
1. Probability Criteria
2. Mass criteria 
3. Impact criteria 
4. Momentum criteria 
5. Others

Do sensitivity analysis on the 1st set of parameters(ie. probability criteria), use the parameter_list variable, obtain thresholds from the graphs. Edit the variables parameters_to_apply_threshold, parameter_thresholds. Edit parameter_list to the 2nd set of variables. Rerun the code, which now carries out sensitivity analysis with the thresholds applied. Obtain thresholds for the 2nd set of variables. Repeat.


Line 95: change datafolder_path to the path on your computer 

Line 116: parameter_list is the contains the parameters to analyse at current stage

Line 119: parameters_to_apply_threshold should contain parameters to apply thresholds on, obtained from previous runs. 

Line 121: parameter_thresholds contains a threshold array. 
	Rows: correspond to parameters in parameters_to_apply_threshold
	Columns: correspond to each background, in the order
['jpsi.csv', 'jpsi_mu_k_swap.csv', 'jpsi_mu_pi_swap.csv', 'k_pi_swap.csv', 
 'phimumu.csv', 'pKmumu_piTok_kTop.csv', 'pKmumu_piTop.csv','psi2S.csv']
	Set value to 0 if no filtering required. 
	
	**Please add more rows as you get to later stages of the analysis.

Line 124: opposite_list_orig has the same shape as parameter_thresholds. Denotes which side of the threshold to discard the data. Default to False -> data below the threshold to be discarded. 

