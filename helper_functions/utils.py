### This file is a collection of functions to aggregate the EV, WLU, and DMREU
### for several causes (vectors of simulated outputs). 

import numpy as np
import pandas as pd
import risk_models.wlu_function as wlu
import risk_models.dmreu_function as dmreu
import copy 
import os


def dmreu_calibration(p):
    '''
    For a gamble where you're guaranteed X or 100X with probability p,
        calculate the risk-aversion power of the DMREU function that corresponds to that probability.
    '''
    c = -2/np.log10(p)
    return c

## Default WLU and DMREU risk aversion levels
default_dmreu_powers = [('p = 0.01', dmreu_calibration(0.01)),
                ('p = 0.02', dmreu_calibration(0.02)), 
                ('p = 0.03', dmreu_calibration(0.03)),
                ('p = 0.04', dmreu_calibration(0.04)),
                ('p = 0.05', dmreu_calibration(0.05)),
                ('p = 0.07', dmreu_calibration(0.07)),
               ('p = 0.10', dmreu_calibration(0.10)),]

default_wlu_powers = [('c = 0', 0), 
                ('c = 0.01', 0.01),
                ('c = 0.05', 0.05), 
                ('c = 0.10', 0.10),
               ('c = 0.15', 0.15),
               ('c = 0.25', 0.25) 
               ]

def describe_xrisk_scenarios(xrisk_scenarios_dict, outcomes_dict, xrisk_causes, output_str, folder = 'results'):
    '''
    Create and return a dataframe with the parameters for the xrisk intervention scenario. 
        Also add the expected value and confidence interval 
    '''
    p_effect = []
    p_good_if_effect = []
    extent_of_bad_effect = []
    bp_risk_reduced = [] 
    avg_risk_reduced = []

    e_value = []
    p_pos = []
    p_neg = []

    for scenario in xrisk_causes:
        p_effect.append(xrisk_scenarios_dict[scenario]['params']['p_has_effect'])
        p_good_if_effect.append(xrisk_scenarios_dict[scenario]['params']['p_good_if_effect'])

        extent_str = str(xrisk_scenarios_dict[scenario]['params']['low_extent_of_negative_effect']) + " to " + str(xrisk_scenarios_dict[scenario]['params']['high_extent_of_negative_effect'])
        extent_of_bad_effect.append(extent_str)

        bp_risk_str = str(xrisk_scenarios_dict[scenario]['params']['low_amt_risk_reduced_per_B']*10000) + " to " + str(xrisk_scenarios_dict[scenario]['params']['high_amt_risk_reduced_per_B']*10000)
        bp_risk_reduced.append(bp_risk_str)

        avg_risk_i = np.round(xrisk_scenarios_dict[scenario]['avg_risk_reduction']*10000, 2)
        avg_risk_reduced.append(avg_risk_i)

        e_value.append(np.mean(outcomes_dict[scenario]))

        p_pos.append(str(xrisk_scenarios_dict[scenario]['p_save_world']*100)+ "%")
        p_neg.append(str(xrisk_scenarios_dict[scenario]['p_end_world']*100)+ "%")

    params_df = pd.DataFrame(list(zip(p_effect, p_good_if_effect, extent_of_bad_effect, bp_risk_reduced, \
                                     avg_risk_reduced)), 
                            index=xrisk_causes, columns=['P(Effect)', 'P(Good | Effect)', 
                                                         'Rel. Magnitide of Bad Effect', 'BP Risk Reduced per $1B', 
                                                         'E(BP Reduced/$1B)'])
    
    outputs_df = pd.DataFrame(list(zip(e_value, p_pos, p_neg)), index=xrisk_causes,
                              columns=['Mean Value, DALYs/$1000', 
                                        'Chance Prevent X-Risk, %', 'Chance Cause X-Risk, %'])

    params_df.to_csv(os.path.join(folder, '{}_params_xrisk_scenarios.csv'.format(output_str)))
    outputs_df.to_csv(os.path.join(folder, '{}_results_xrisk_scenarios.csv'.format(output_str)))

    return params_df, outputs_df

def add_user_wlu_power(user_wlu):
    '''
    Given a user's WLU coefficient, add it to the default WLU powers.
    '''
    copy_default_wlu_powers = copy.deepcopy(default_wlu_powers)
    if ('c = {}'.format(user_wlu), user_wlu) not in copy_default_wlu_powers:
        copy_default_wlu_powers.append(('c = {}'.format(user_wlu), user_wlu))
    return copy_default_wlu_powers

def add_user_dmreu_power(user_prob):
    '''
    Given a user's probability of choosing a risky gamble over a guaranteed payoff, 
        calculate the power of the DMREU function that corresponds to that probability. 
        Add the power to the default DMREU weights.
    '''
    copy_default_dmraev_powers =copy.deepcopy(default_dmreu_powers)
    if ('p = {}'.format(str(user_prob)), dmreu_calibration(user_prob)) not in copy_default_dmraev_powers:
        copy_default_dmraev_powers.append(('p = {}'.format(str(user_prob)), dmreu_calibration(user_prob)))
    return copy_default_dmraev_powers

def get_ev_and_percentiles(outputs_dict, causes): 
    '''
    Takes a dictionary of vectors of the cost-effectivenesses for all causes assessed, and 
        return a tuple of summary statistics for the DALYs/$1000 (5th, 25th, 50th, 75th, 95th-percentiles + the mean)
    '''
    evs = []
    point_15_pct = []
    one_pct = []
    fifth_pct = []
    twentyfifth_pct = []
    median = []
    seventyfifth_pct = []
    ninetyfifth_pct = []
    ninetyninth_pct = []
    ninetynine_point_85_pct = []

    for cause in causes: 
        outputs = outputs_dict[cause]
        evs.append(np.round(np.mean(outputs),2))
        point_15_pct.append(np.round(np.percentile(outputs, 0.15),2))
        one_pct.append(np.round(np.percentile(outputs, 1),2))
        fifth_pct.append(np.round(np.percentile(outputs, 5),2))
        twentyfifth_pct.append(np.round(np.percentile(outputs, 25),2))
        median.append(np.round(np.percentile(outputs, 50),2))
        seventyfifth_pct.append(np.round(np.percentile(outputs, 75),2))
        ninetyfifth_pct.append(np.round(np.percentile(outputs, 95),2))
        ninetyninth_pct.append(np.round(np.percentile(outputs, 99),2))
        ninetynine_point_85_pct.append(np.round(np.percentile(outputs, 99.85),2))

    evs = np.array(evs)
    point_15_pct = np.array(point_15_pct)
    one_pct = np.array(one_pct)
    fifth_pct = np.array(fifth_pct)
    twentyfifth_pct = np.array(twentyfifth_pct)
    median = np.array(median)
    seventyfifth_pct = np.array(seventyfifth_pct)
    ninetyfifth_pct = np.array(ninetyfifth_pct)
    ninetyninth_pct = np.array(ninetyninth_pct)
    ninetynine_point_85_pct = np.array(ninetynine_point_85_pct)

    return evs, point_15_pct, one_pct, fifth_pct, twentyfifth_pct, median, \
        seventyfifth_pct, ninetyfifth_pct, ninetyninth_pct, ninetynine_point_85_pct

def get_wlus(outputs_dict, causes, user_wlu, money_spent, baseline_utility = [], to_print = False):
    '''
    Takes a dictionary of vectors of the cost-effectivenesses for all the causes assessed 
        and the set of WLU risk-aversion levels. Calculates the weighted linear utility for 
        each cause at each risk-aversion level. Returns a dictionary of vectors of WLUs for
        each cause at each risk-aversion level.
    '''
    wlu_powers = add_user_wlu_power(user_wlu)
    wlus_by_power = {}

    multiplier_money = money_spent/1000

    for power_tuple in wlu_powers:
        power_str = power_tuple[0]
        power = power_tuple[1]
        wlu_per_1000 = []
        for cause in causes:
            dalys_per_1000 = 0
            dalys_per_1000 = outputs_dict[cause]
            if to_print:
                print("DALYs per 1000 for {}: {}".format(cause, dalys_per_1000))
            dalys_total = dalys_per_1000 * multiplier_money
            if to_print:
                print("DALYs total for {}: {}".format(cause, dalys_total))
            if baseline_utility != []:
                dalys_total += np.array(baseline_utility)
            wlu_total_cause = round(wlu.get_weighted_linear_utility(dalys_total, power = power),2)
            if to_print:
                print("WLU for {}: {}".format(cause, wlu_total_cause))
            wlu_per_1000_cause = wlu_total_cause / multiplier_money
            if to_print:
                print("WLU per 1000 for {}: {}".format(cause, wlu_per_1000_cause))
            wlu_per_1000.append(wlu_per_1000_cause)
        wlus_per_1000 = np.array(wlu_per_1000)
        wlus_by_power[power_str] = wlus_per_1000
    if to_print:
        print("WLUs per 1000: {}".format(wlus_by_power))

    return wlus_by_power

def get_dmreus(outputs_dict, causes, user_power):
    '''
    Takes a dictionary of vectors of the cost-effectivenesses for all the causes assessed and
        the set of REU risk-aversion levels. Calculates the risk-weighted utility for each cause
        at each risk-aversion level. Returns a dictionary of vectors of REUs for each cause at
        each risk-aversion level.
    '''
    
    dmreu_powers = add_user_dmreu_power(user_power)
    dmreus_by_power = {}

    for prob_tuple in dmreu_powers:
        prob = prob_tuple[0]
        power = prob_tuple[1]
        reus = []
        for cause in causes:
            outputs = outputs_dict[cause]
            reus.append(round(dmreu.get_risk_weighted_utility(outputs, a = power),2))
        reus = np.array(reus)
        dmreus_by_power[prob] = reus

    return dmreus_by_power

def make_ev_dataframe(outputs_dict, causes, path, folder='results'):
    '''
    Make a dataframe containting the cost-effectiveness (DALYs/$1000) for each cause 
        and create a csv file of the dataframe.
    '''

    evs, point_15_pct, one_pct, fifth_pct, twentyfifth_pct, median, \
        seventyfifth_pct, ninetyfifth_pct, ninetyninth_pct, ninetynine_point_85_pct = get_ev_and_percentiles(outputs_dict, causes)

    ev_df = pd.DataFrame({'Mean DALYs/$1000': evs,
                            '0.15th-pct': point_15_pct,
                            '1st-pct': one_pct,
                            '5th-pct': fifth_pct,
                            '25th-pct': twentyfifth_pct,
                            '50th-pct': median,
                            '75th-pct': seventyfifth_pct,
                            '95th-pct': ninetyfifth_pct,
                            '99th-pct': ninetyninth_pct,
                            '99.85th-pct': ninetynine_point_85_pct}, 
                            index=causes)
    ev_df.to_csv(os.path.join(folder, '{}_expected_value_results.csv'.format(path)))
    
    return ev_df

def make_wlu_dataframe(outputs_dict, causes, user_wlu, path, money_spent, folder = 'results', baseline_utility = [], to_print = False):
    '''
    Make a dataframe containting the weighted linear utility for each cause at each risk-aversion level
        and create a csv file of the dataframe.
    Baseline utility is not used because we're assuming neutrality 
    '''

    wlus_by_power = get_wlus(outputs_dict, causes, user_wlu, money_spent, baseline_utility, to_print=to_print)
    wlu_per_money_spent = wlus_by_power
    powers_lst = [power_tuple[0] for power_tuple in add_user_wlu_power(user_wlu)]

    wlu_df = pd.DataFrame(wlus_by_power, index=causes, columns=powers_lst)
    wlu_df.to_csv(os.path.join(folder, '{}_weighted_linear_utility_results.csv'.format(path)))

    return wlu_df

def make_dmreu_dataframe(outputs_dict, causes, user_power, path, folder = 'results'):
    '''
    Make a dataframe containting the DMREU for each cause at each risk-aversion level
        and create a csv file of the dataframe.
    '''
    
    dmreus_by_power = get_dmreus(outputs_dict, causes, user_power)
    probs_lst = [power_tuple[0] for power_tuple in add_user_dmreu_power(user_power)]

    dmraev_df = pd.DataFrame(dmreus_by_power, index=causes, columns=probs_lst)
    dmraev_df.to_csv(os.path.join(folder, '{}_DMREU_results.csv'.format(path)))

    return dmraev_df

