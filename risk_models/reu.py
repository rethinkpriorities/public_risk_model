### REU functions

import numpy as np
import pandas as pd
import squigglepy as sq
from squigglepy.numbers import K,M,B
import copy
import os
import risk_models.ambiguity_aversion as aa

interventions = ["Nothing", "AMF", "CF campaign", "Shrimp welfare - stunning", "Shrimp welfare - NH3", "Conservative x-risk work", "Risky x-risk work"]

N = 8*M

MONEY = 100*M

MULTIPLES_OF_1K = MONEY/1000
MULTIPLES_OF_1B = MONEY/(1*B)

def dmraev_calibration(p):
    '''
    For a gamble where you're guaranteed X or 100X with probability p,
        calculate the risk-aversion power of the DMRAEV function that corresponds to that probability.
    '''
    c = -2/np.log10(p)
    return c

def risk_function(a, p):
    '''
    Weighting the probability of getting at least X by the risk-aversion coefficient
    '''
    r_p = p**a
    return r_p

def event_probabilities_dict():
    '''
    Generate the probability of each of the factors that determine the state of the world. 
        
    These include the probability of an xrisk occurring in the next 100 years, 
        the probability that chickens are sentient, and the probability that shrimp are sentient. 
    '''
    probs = {
        'P(x-risk occurs)': sq.sample(sq.lognorm(0.05,0.25, lclip=0.01, rclip=0.5), N),
        'P(chickens sentient)': np.ones(N) - sq.sample(sq.lognorm(0.05, 0.25, lclip=0.01, rclip=0.5), N),
        'P(shrimp sentient)': sq.sample(sq.lognorm(0.2, 0.7, lclip = 0.01, rclip=1), N),
    }
    return probs

def create_states_of_world_dict():
    '''
    Create a dictionary of all the states of the world that are under consideration.
        There are eight based on the binary outcomes for whether an xrisk occurs in the next
        century, whether chickens are sentient, and whether shrimp are sentient.
    '''
    states_dict = {"X/C/S": {
                           "x-risk occurs": 1, 
                           "chickens sentient": 1, 
                           "shrimp sentient": 1}, 
                "X/C/Sc": {
                           "x-risk occurs": 1, 
                           "chickens sentient": 1, 
                           "shrimp sentient": 0}, 
                "X/Cc/S": {
                            "x-risk occurs": 1,
                            "chickens sentient": 0,
                            "shrimp sentient": 1},
                "X/Cc/Sc": {
                           "x-risk occurs": 1, 
                           "chickens sentient": 0, 
                           "shrimp sentient": 0}, 
                "Xc/C/S": {
                           "x-risk occurs": 0, 
                           "chickens sentient": 1, 
                           "shrimp sentient": 1}, 
                "Xc/C/Sc": {
                           "x-risk occurs": 0, 
                           "chickens sentient": 1, 
                           "shrimp sentient": 0}, 
                "Xc/Cc/S": {
                           "x-risk occurs": 0, 
                           "chickens sentient": 0, 
                           "shrimp sentient": 1}, 
                "Xc/Cc/Sc": {
                           "x-risk occurs": 0, 
                           "chickens sentient": 0, 
                           "shrimp sentient": 0}}
    return states_dict

def create_states_of_world_df(states_dict):
    '''
    Create a dataframe of all the states of the world that are under consideration.
    This is just to check that the dictionary is correct.
    '''
    states_df = pd.DataFrame.from_dict(states_dict).T
    return states_df

def make_daly_burdens_by_harm_dict(xrisk_dalys_at_stake, shrimp_slaughter_human_daly_burden, shrimp_nh3_human_daly_burden, chicken_human_daly_burden, to_print=False):
    '''
    Get a dictionary of the global DALY burden for each of the harms. These are annualized for 
        malaria, chickens, and shrimp, and total for xrisk (the future that we 
        could have a counterfactual impact over). 
    For the shrimp and chickens, the DALY burdens are adjusted for the sentience-conditioned 
        welfare ranges of the animals. 
    '''
    daly_burdens_by_harm = {'malaria': -1*sq.sample(sq.norm(mean=63*M, sd=5*M), N), #https://ourworldindata.org/burden-of-disease#the-disease-burden-by-cause
                    'x-risk': -1*xrisk_dalys_at_stake,
                    'chickens': -1*chicken_human_daly_burden,
                    'shrimp - slaughter': -1*shrimp_slaughter_human_daly_burden,  
                    'shrimp - NH3': -1*shrimp_nh3_human_daly_burden}      

    if to_print:
        print("DALY burden by harm: {}".format(daly_burdens_by_harm))
    return daly_burdens_by_harm

def get_daly_burden_by_state(states_dict, state, daly_burden_by_harm, to_print=False):
    '''
    This function takes one of the eight states of the world and calculates the 
        DALY burden of this state 
    '''
    
    xrisk = states_dict[state]["x-risk occurs"]
    chickens = states_dict[state]["chickens sentient"]
    shrimp = states_dict[state]["shrimp sentient"]

    daly_burden_by_state = 0
    if to_print:
        print(daly_burden_by_harm)
        print("START DALY burden for state: {}".format(daly_burden_by_state))
        print("DALY burden malaria: {}".format(daly_burden_by_harm['malaria']))

    #  add in malaria because we know children are dying from it 
    daly_burden_by_state += daly_burden_by_harm['malaria']

    if to_print:
        print("daly_burden_by_state (added malaria): {}".format(daly_burden_by_state))
    if xrisk == 1:
        if to_print:
            print("DALY burden xrisk: {}".format(daly_burden_by_harm['x-risk']))
        daly_burden_by_state += daly_burden_by_harm['x-risk']
        if to_print:
            print("daly_burden_by_state (added xrisk): {}".format(daly_burden_by_state))
    if chickens == 1:
        daly_burden_by_state += daly_burden_by_harm['chickens']
        if to_print:
            print("daly_burden_by_state (added chickens): {}".format(daly_burden_by_state))
    if shrimp == 1:
        daly_burden_by_state += daly_burden_by_harm['shrimp - slaughter']
        if to_print:
            print("daly_burden_by_state (added shrimp - slaughter): {}".format(daly_burden_by_state))
        daly_burden_by_state += daly_burden_by_harm['shrimp - NH3']
        if to_print:
            print("daly_burden_by_state (added shrimp - NH3): {}".format(daly_burden_by_state))
    if to_print:
        print("END DALY burden for state: {}".format(daly_burden_by_state))
    
    return daly_burden_by_state

def make_daly_burdens_dict(states_dict, daly_burden_by_harm, to_print=False):
    '''
    Makes a dictionary of all the DALY burdens of each of the eight states
    '''
    daly_burden_dict = {
        'X/C/S': get_daly_burden_by_state(states_dict, 'X/C/S', daly_burden_by_harm, to_print),
        'X/C/Sc': get_daly_burden_by_state(states_dict, 'X/C/Sc', daly_burden_by_harm, to_print),
        'X/Cc/S': get_daly_burden_by_state(states_dict, 'X/Cc/S', daly_burden_by_harm, to_print),
        'X/Cc/Sc': get_daly_burden_by_state(states_dict, 'X/Cc/Sc', daly_burden_by_harm, to_print),
        'Xc/C/S': get_daly_burden_by_state(states_dict, 'Xc/C/S', daly_burden_by_harm, to_print),
        'Xc/C/Sc': get_daly_burden_by_state(states_dict, 'Xc/C/Sc', daly_burden_by_harm, to_print),
        'Xc/Cc/S': get_daly_burden_by_state(states_dict, 'Xc/Cc/S', daly_burden_by_harm, to_print),
        'Xc/Cc/Sc': get_daly_burden_by_state(states_dict, 'Xc/Cc/Sc', daly_burden_by_harm, to_print),
    }

    if to_print:
        print("DALY burden dict: {}".format(daly_burden_dict))

    return daly_burden_dict

def get_joint_prob_state(p_xrisk, p_chicken_sent, p_shrimp_sent, chicken_sent, shrimp_sent, xrisk_occurs):
    '''
    Get the joint probability of each state of the world given the probabilities for 
        chicken and shrimp sentience and the proability of xrisk
    '''
    
    joint_prob = 1
    if xrisk_occurs == 1:
        joint_prob *= p_xrisk
    elif xrisk_occurs == 0:
        joint_prob *= (1 - p_xrisk)
    if chicken_sent == 1:
        joint_prob *= p_chicken_sent
    elif chicken_sent == 0: 
        joint_prob *= (1 - p_chicken_sent)
    if shrimp_sent == 1:
        joint_prob *= p_shrimp_sent
    elif shrimp_sent == 0:   
        joint_prob *= (1 - p_shrimp_sent)

    return joint_prob
    
def make_joint_prob_states_dict(states_dict, event_probs):
    '''
    Make a dictionary with the joint probability of each of the eight 
        states of the world coming about. 
    '''

    joint_probability_dict = {}
    for state in states_dict:
        p_xrisk = event_probs["P(x-risk occurs)"]
        p_chicken_sent = event_probs["P(chickens sentient)"]
        p_shrimp_sent = event_probs["P(shrimp sentient)"]

        xrisk_occurs = states_dict[state]["x-risk occurs"]
        chickens_sent = states_dict[state]["chickens sentient"]
        shrimp_sent = states_dict[state]["shrimp sentient"]  

        joint_probability_dict[state] = get_joint_prob_state(p_xrisk, p_chicken_sent, p_shrimp_sent, chickens_sent, shrimp_sent, xrisk_occurs)
    return joint_probability_dict

def get_dalys_saved_spend_money(intervention, chicken_sent, shrimp_sent, amf_dalys_per_1000, hens_dalys_per_1000, \
                             shrimp_slaughter_dalys_per_1000, shrimp_nh3_dalys_per_1000, daly_burden_by_harm):
    '''
    Estimate the number of DALYs averted through spending $1B on an intervention, 
        given the intervention and the state of the world. This only applies to 
        the interventions that directly reduce the DALY burden of the world, not the
        probability of each state coming about. 
    '''
    
    if intervention == "Nothing":
        effect = np.zeros(N)
    elif intervention == "AMF":
        malaria_burden_pos = -1*daly_burden_by_harm['malaria']
        effect = np.array([min(amf_dalys_per_1000[i]*MULTIPLES_OF_1K, malaria_burden_pos[i]) for i in range(N)])
    elif intervention == "CF campaign":
        if chicken_sent == 1:
            chicken_daly_burden_pos = -1*daly_burden_by_harm['chickens']
            effect = np.array([min(hens_dalys_per_1000[i]*MULTIPLES_OF_1K, chicken_daly_burden_pos[i]) for i in range(N)])
        else: 
            effect = np.zeros(N)
    elif intervention == "Shrimp welfare - stunning":
        if shrimp_sent == 1:
            shrimp_slaughter_daly_burden_pos = -1*daly_burden_by_harm['shrimp - slaughter']
            effect = np.array([min(shrimp_slaughter_dalys_per_1000[i]*MULTIPLES_OF_1K, shrimp_slaughter_daly_burden_pos[i]) for i in range(N)])
        else:
            effect = np.zeros(N)
    elif intervention == "Shrimp welfare - NH3":
        if shrimp_sent == 1:
            shrimp_nh3_daly_burden_pos = -1*daly_burden_by_harm['shrimp - NH3']
            effect = np.array([min(shrimp_nh3_dalys_per_1000[i]*MULTIPLES_OF_1K, shrimp_nh3_daly_burden_pos[i]) for i in range(N)])
        else:
            effect = np.zeros(N)
    return effect

def get_decrease_xrisk_money(risk_reduction_per_bn_vector, to_print=False):
    '''
    Return the risk reduction for the $1B spent on x-risk work and print it if we want
    '''
    decrease_xrisk = risk_reduction_per_bn_vector*MULTIPLES_OF_1B
    if to_print:
        print("Decrease in P(x-risk) per ${} Spent: {}".format(MONEY, decrease_xrisk))

    return decrease_xrisk

def get_new_xrisk(idx, baseline_p_xrisk, decrease_xrisk):
    candidate_new_p_xrisk = baseline_p_xrisk[idx] - decrease_xrisk[idx]
    if candidate_new_p_xrisk > 0.0000001:
        return candidate_new_p_xrisk
    else:
        return 0.0000001

def get_lottery_for_action(intervention, daly_burden_by_harm, states_dict, daly_burden_dict, joint_prob_dict, events_prob_dict, \
                           amf_dalys_per_1000, hens_dalys_per_1000, shrimp_slaughter_dalys_per_1000, shrimp_nh3_dalys_per_1000, \
                           conservative_xrisk_reduced_per_bn, risky_xrisk_reduced_per_bn, to_print=False):
    '''
    Define the lottery for a single intervention. This is comprised with a (utility, probability)
        pair for the eight states of the world. For the interventions that directly reduce the DALY burden, 
        I calculate the change in utility of each state and add it to the baseline utility for that state. 
    For the xrisk interventions, I calculate the reduction in risk and add that to the baseline risk to get the 
        new joint probabilities of each state occurring. 
    '''
    
    lottery = {}
    lottery_old = {}

    if intervention in ["Nothing", "AMF", "CF campaign", "Shrimp welfare - stunning", "Shrimp welfare - NH3"]:
        for state in states_dict.keys():
            chicken_sent = states_dict[state]["chickens sentient"]
            shrimp_sent = states_dict[state]["shrimp sentient"]
            xrisk_occurs = states_dict[state]["x-risk occurs"]

            baseline_dalys = daly_burden_dict[state]
            change_dalys = get_dalys_saved_spend_money(intervention, chicken_sent, shrimp_sent, amf_dalys_per_1000, \
                                                     hens_dalys_per_1000, shrimp_slaughter_dalys_per_1000, shrimp_nh3_dalys_per_1000, \
                                                    daly_burden_by_harm)

            lottery[state] = {"daly burden": baseline_dalys + change_dalys,
                        "joint probability": joint_prob_dict[state]}
            
            lottery_old[state] = {"daly burden": baseline_dalys,
                        "joint probability": joint_prob_dict[state]}
            if to_print:    
                print("state: {}".format(state))
                print("Intervention: {}".format(intervention))
                print("Baseline DALYs: {}".format(baseline_dalys))
                print("Change in DALYs: {}".format(change_dalys))
                print("New DALYs: {}".format(lottery[state]["daly burden"]))
            
    elif intervention in ["Conservative x-risk work", "Risky x-risk work"]:
        baseline_p_xrisk = events_prob_dict["P(x-risk occurs)"]
        p_chicken_sent = events_prob_dict["P(chickens sentient)"]
        p_shrimp_sent = events_prob_dict["P(shrimp sentient)"]
        
        if intervention == "Conservative x-risk work":
            decrease_xrisk = get_decrease_xrisk_money(conservative_xrisk_reduced_per_bn, to_print)
        elif intervention == "Risky x-risk work":
            decrease_xrisk = get_decrease_xrisk_money(risky_xrisk_reduced_per_bn, to_print)
        new_p_xrisk = []

        for i in range(N):
            new_p_xrisk.append(get_new_xrisk(i, baseline_p_xrisk, decrease_xrisk))
        new_p_xrisk = np.array(new_p_xrisk)
        for state in states_dict.keys():
            chicken_sent = states_dict[state]["chickens sentient"]
            shrimp_sent = states_dict[state]["shrimp sentient"]
            xrisk_occurs = states_dict[state]["x-risk occurs"]

            old_joint_prob = joint_prob_dict[state]
            new_joint_prob = get_joint_prob_state(new_p_xrisk, p_chicken_sent, p_shrimp_sent, chicken_sent, shrimp_sent, xrisk_occurs)

            lottery[state] = {"daly burden": daly_burden_dict[state],
                            "joint probability": new_joint_prob}
            lottery_old[state] = {"daly burden": daly_burden_dict[state],
                            "joint probability": joint_prob_dict[state]}
            
            daly_burden = daly_burden_dict[state]
            
            if to_print:
                print("state: {}".format(state))
                print("Intervention: {}".format(intervention))
                print("Mean Baseline P(X-risk): {}".format(np.mean(baseline_p_xrisk)))
                print("Mean Decrease P(X-risk): {}".format(np.mean(decrease_xrisk)))
                print("Mean New P(X-risk): {}".format(np.mean(new_p_xrisk)))
                print("xrisk occurs: {}".format(xrisk_occurs))
                print("chickens sent: {}".format(chicken_sent))
                print("shrimp sent: {}".format(shrimp_sent))
                print("Mean P(Chickens sentient): {}".format(np.mean(p_chicken_sent)))
                print("Mean P(Shrimp sentient): {}".format(np.mean(p_shrimp_sent)))
                print("Mean Old P(state): {}".format(np.mean(old_joint_prob)))
                print("Mean New P(state): {}".format(np.mean(new_joint_prob)))
                print("Baseline DALYs: {}".format(daly_burden))
                print("Lottery: {}".format(lottery))

    return lottery

def get_ranked_state_tuples_one_sim(idx, lottery, to_print=False):
    '''
    For a single simulation (one of 2M), I rank the DALY burdens 
        of each state of the world from the worst outcome to best. 

    The function creates a list of eight tuples, each with the state, the DALY burden, 
        and the joint probability of that state occurring. 
    '''
    state_utility_prob_tuples = []
    for state in lottery.keys():
        state_utility_prob_tuples.append((state, lottery[state]["daly burden"][idx], lottery[state]["joint probability"][idx]))
    ranked_state_utility_prob_tuples = sorted(state_utility_prob_tuples, key=lambda x: x[1], reverse=False)

    if to_print:
        print("Worst to best")
        print(ranked_state_utility_prob_tuples)
    return ranked_state_utility_prob_tuples

def get_probability_of_at_least_as_good_one_sim(ranked_state_utility_probs):
    '''
    For one of the 2M simulations, I take the ranked list of state tuples. For each state,
        I calculate the probability of landing in a state that is at least as good 
        (lower DALY burden) as that state. 
    The result is a list of eight tuples with the state, the DALY burden, and the probability
        of getting a state that is at least as good as that state.
    '''
    
    ranked_state_utility_cumulative_probs = []
    for i in range(len(ranked_state_utility_probs)):
        probability_of_at_least_as_good = 0
        for j in range(i, len(ranked_state_utility_probs)):
            probability_of_at_least_as_good += ranked_state_utility_probs[j][2]
        ranked_state_utility_cumulative_probs.append((ranked_state_utility_probs[i][0], ranked_state_utility_probs[i][1], probability_of_at_least_as_good))

    return ranked_state_utility_cumulative_probs

def get_reu_one_sim(ranked_state_utility_cumulative_probs, a, to_print=False):
    '''
    For one simulation of 2M, I calculate the risk-averse expected utility 
        of the lottery, given the risk-aversion coefficient. This 
        1) weights the probability of getting states at least as good as each state with the risk function
        2) for each state, calculates the change in the DALY burden from going from one state to the next-best state
        3) multiplies together this weighted probability and the change in DALY burden for each state and the next-best state
        4) adds up all these risk-weighted changes across all states in DALY burden to get the risk-averse expected utility
            for that simulation. 
    '''
    
    reu = 0
    for i, tuple in enumerate(ranked_state_utility_cumulative_probs):
        probability_i = tuple[2]
        r_p = risk_function(a, probability_i)
        if i == 0:
            utility_i = tuple[1]
            reu += r_p*utility_i
            if to_print:
                print("prob at least: ", r_p)
                print("utility_i:", utility_i)
                print("contribution:", r_p*utility_i)
        else:
            utility_i_minus_1 = ranked_state_utility_cumulative_probs[i-1][1]
            utility_i = tuple[1]
            reu += r_p*(utility_i - utility_i_minus_1)
            if to_print:
                print("prob at least: ", r_p)
                print("utility_i:", utility_i)
                print("utility_i_minus_1:", utility_i_minus_1)
                print("utility_i - utility_i_minus_1:", utility_i - utility_i_minus_1)
                print("contribution:", r_p*(utility_i - utility_i_minus_1))
        if to_print:
            print("reu:", reu)
    return reu

def get_reu_given_lottery_one_idx(idx, lottery, a, to_print=False):
    '''
    For a given index and a given action, I calculate the risk-averse expected utility. The lottery defines
        the possible outcomes for taking that action based on the state of the world. 
    '''
    ranked_state_utility_probs = get_ranked_state_tuples_one_sim(idx, lottery, to_print=False)
    ranked_state_utility_cumulative_probs = get_probability_of_at_least_as_good_one_sim(ranked_state_utility_probs)
    reu_i = get_reu_one_sim(ranked_state_utility_cumulative_probs, a)

    if to_print:
        print("Index: {}".format(idx))
        print("Ranked state utility probs: {}".format(ranked_state_utility_probs))
        print("Ranked state utility cumulative probs: {}".format(ranked_state_utility_cumulative_probs))
        print("REU: {}".format(reu_i))

    return reu_i

def get_one_action_reus(a, action, lottery, to_print=False):
    '''
    For one action: define the possible payoffs, and calculate the risk-averse expected utility for each of the 2M simulations.
    '''
    reus_action = []

    for idx in range(N):
        reu_a_i = get_reu_given_lottery_one_idx(idx, lottery, a, to_print)
        reus_action.append(reu_a_i)

    reus_action = np.array(reus_action)

    if to_print:
        print("Action: {}".format(action))
        print("Lottery: {}".format(lottery))
        print("REUs for action {}: {}".format(action, reus_action))

    return reus_action

def get_reus_all_actions(a, daly_burden_by_harm, states_dict, daly_burdens_dict, joint_probs_dict, event_probs, \
                        amf_dalys_per_1000, hens_dalys_per_1000, shrimp_slaughter_dalys_per_1000, shrimp_nh3_dalys_per_1000, \
                        conservative_xrisk_reduction_per_bn, risky_xrisk_reduction_per_bn, to_print):
    '''
    For each of the 2M simulations, I calculate the risk-averse expected utility of each of the interventions. 
    
    The result is a vector of the REU values of spending $1B on each intervention, organized
        into a dictionary of the interventions and their REU vector. 
    
    The change in REU for each simulation from doing nothing is also calculated and organized into a dictionary.
    '''
    reus = {}
    changes_in_reu = {}

    for action in interventions:
        lottery = get_lottery_for_action(action, daly_burden_by_harm, states_dict, daly_burdens_dict, joint_probs_dict, event_probs, \
                                amf_dalys_per_1000, hens_dalys_per_1000, shrimp_slaughter_dalys_per_1000, shrimp_nh3_dalys_per_1000, \
                                conservative_xrisk_reduction_per_bn, risky_xrisk_reduction_per_bn, to_print)
        
        reus_action = get_one_action_reus(a, action, lottery, to_print)
        reus[action] = reus_action
        if action != "Nothing":
            changes_in_reu[action] = reus_action - reus["Nothing"]
        else:
            changes_in_reu[action] = np.zeros(N)
            
    return reus, changes_in_reu

def make_summary_stats_reu_df(reus, a, output_name):
    '''
    Making a dataframe with summary states for the distribution of REU values, 
        for a specified risk-aversion coefficient.

    This incorporates four ways of aggregating the REU values under different 
        levels of ambiguity aversion. 
    '''
    mean = []
    aa_ev_4 = get_ambiguity_aversion_weighted_utility(reus, 4, aa.cubic_weighting)
    aa_ev_8 = get_ambiguity_aversion_weighted_utility(reus, 8, aa.cubic_weighting)
    fifth_percentile = []
    ninety_fifth_percentile = []
    median = []
    cols = ["Mean, a = {}".format(round(a,2)), "Ambiguity averse, 1.5x weight to worst", "Ambiguity averse, 2x weight to worst", \
             "fifth percentile", "ninety fifth percentile", "median"]
    idx = reus.keys()

    for action in reus.keys():
        mean.append(np.mean(reus[action]))
        fifth_percentile.append(np.percentile(reus[action], 5))
        ninety_fifth_percentile.append(np.percentile(reus[action], 95))
        median.append(np.percentile(reus[action], 50))
    reu_df = pd.DataFrame(list(zip(mean, aa_ev_4, aa_ev_8, fifth_percentile, ninety_fifth_percentile, median)), \
                          columns=cols, index=idx)
    sorted_reu_df = reu_df.sort_values(by=["Mean, a = {}".format(round(a,2))], ascending=False)
    sorted_reu_df.to_csv(os.path.join('results', output_name))

    return sorted_reu_df

def get_ratio_of_change_in_reu_by_cause(changes_in_reu, output_name):
    '''
    This function allows you to take the changes in REU over doing nothing 
        for each intervention, and it makes a table of the ratio of the average 
        REU change for the row intervention to the average REU change for the column
        intervention. 
    '''
    ratio_of_avg_changes_in_reu_by_cause = {}
    causes = ["AMF", "CF campaign", "Shrimp welfare - stunning", "Shrimp welfare - NH3", "Conservative x-risk work", "Risky x-risk work"]
    for row_intervention in causes:
        ratio_of_avg_changes_in_reu_by_cause[row_intervention] = {}
        row_int_reu_change = changes_in_reu[row_intervention]
        for col_action in causes:
            col_int_reu_change = changes_in_reu[col_action]
            ratio_of_avg_changes_in_reu_by_cause[row_intervention][col_action] = np.mean(row_int_reu_change)/np.mean(col_int_reu_change)
    df_ratio_of_change_in_reu_by_cause = pd.DataFrame.from_dict(ratio_of_avg_changes_in_reu_by_cause, orient='index')
    df_ratio_of_change_in_reu_by_cause.to_csv(os.path.join('results', output_name))
    return df_ratio_of_change_in_reu_by_cause

def get_proportion_time_best_action(reus, output_name):
    '''
    A function that takes the dictionary of REU values for all interventions that could
        be pursued and calculates the proportion of time that each intervention has the 
        highest (least negative) REU.
    '''
    count_times_best_action = {a: 0 for a in reus.keys()}
    for i in range(N):
        action_reu_pairs = []
        for action in reus.keys():
            action_reu_pairs.append((action, reus[action][i]))
        best_action = sorted(action_reu_pairs, key=lambda x: x[1], reverse=True)[0][0]
        count_times_best_action[best_action] += 1
    proportion_times_best_action = {a: count_times_best_action[a]/N for a in reus.keys()}
    df_proportion_times_best_action = pd.DataFrame.from_dict(proportion_times_best_action, 
                                                             orient='index', columns=["Proportion of time best action"])
    
    df_proportion_times_best_action.to_csv(os.path.join('results', output_name))
    
    return proportion_times_best_action, df_proportion_times_best_action

def get_ambiguity_aversion_weighted_utility(reus, coef, weighting_function, to_print=False):
    '''
    Takes the dictionary of REU values for all interventions that could be pursued and
        a specified function for aggregating REUs under ambiguity averse preferences

    Then, it takes the REU values for each intervention and calculates the ambiguity-averse
        aggregated value of each intervention.
    '''
    
    aa_ev = []
    for action in reus.keys():
        sorted_reus = np.sort(reus[action])
        aa_ev.append(aa.get_ambiguity_weighted_utility(sorted_reus, coef, weighting_function, to_print))

    return aa_ev

def create_necessary_dictionaries_to_define_lotteries(xrisk_dalys_at_stake, shrimp_slaughter_human_daly_burden, \
                                                      shrimp_nh3_human_daly_burden, chicken_human_daly_burden, to_print=False):
    '''
    Create the dictionaries that define the states of the world, the daly burden of each harm, 
        the probabilities of each event occurring, the daly burden dictionary for all harms, and
        the probability of each state of the world occurring
    '''
    states_dict = create_states_of_world_dict()
    daly_burden_by_harm = make_daly_burdens_by_harm_dict(xrisk_dalys_at_stake, shrimp_slaughter_human_daly_burden, \
                                                         shrimp_nh3_human_daly_burden, chicken_human_daly_burden, to_print)
    event_probs = event_probabilities_dict()

    daly_burdens_dict = make_daly_burdens_dict(states_dict, daly_burden_by_harm, to_print)  
    joint_probs_dict = make_joint_prob_states_dict(states_dict, event_probs)

    return states_dict, daly_burdens_dict, joint_probs_dict, event_probs, daly_burden_by_harm


def reu_main(a, inputs, results_str, to_print=False):
    '''
    Main function that is called from the Jupyter Notebook. 

    The function creates all the dictionaries for the DALY burdens of each state, 
        the joint probability of each state occurring. 

    Then, it calculates the risk-neutral REU values to provide a baseline for comparison
        (and provides ambiguity-averse aggregated values for this risk-neutral baseline).

    Then, it calculates the risk-averse REU values for the specified risk-aversion coefficient
        and provides ambiguity-averse aggregated values for this risk-averse baseline.

    The changes in REU from doing nothing are estimated, as well as the proportion of time
        each intervention has the highest REU and the ratio of the average changes in REU 
        (ambiguity neutral)
        )
    '''
    amf_dalys_per_1000, hens_sc_dalys_per_1000, shrimp_slaughter_sc_dalys_per_1000, shrimp_nh3_sc_dalys_per_1000, \
    shrimp_slaughter_human_daly_burden, shrimp_nh3_human_daly_burden, chicken_human_daly_burden, \
    conservative_xrisk_reduction_per_bn, risky_xrisk_reduction_per_bn, xrisk_dalys_at_stake = inputs

    states_dict, daly_burdens_dict, joint_probs_dict, event_probs, daly_burden_by_harm = create_necessary_dictionaries_to_define_lotteries(xrisk_dalys_at_stake, \
                                                                                                                     shrimp_slaughter_human_daly_burden, \
                                                                                                                     chicken_human_daly_burden, \
                                                                                                                     shrimp_nh3_human_daly_burden, to_print)
    
    # risk neutral
    reus_all_actions_1, changes_in_reu_1 = get_reus_all_actions(1, daly_burden_by_harm, states_dict, daly_burdens_dict, joint_probs_dict, event_probs, \
                                                            amf_dalys_per_1000, hens_sc_dalys_per_1000, shrimp_slaughter_sc_dalys_per_1000, 
                                                            shrimp_nh3_sc_dalys_per_1000, conservative_xrisk_reduction_per_bn, risky_xrisk_reduction_per_bn, to_print)
    reu_df_1 = make_summary_stats_reu_df(reus_all_actions_1, 1, "{}_REU_risk_neutral.csv".format(results_str))
    changes_in_reu_df_1 = make_summary_stats_reu_df(changes_in_reu_1, 1, "{}_changes_in_REU_risk_neutral.csv".format(results_str))
    prop_time_best_action_1, df_prop_time_best_action_1 = get_proportion_time_best_action(reus_all_actions_1, "{}_variance_in_best_action_risk_neutral.csv".format(results_str))
    df_ratio_of_changes_in_reu_1 = get_ratio_of_change_in_reu_by_cause(changes_in_reu_1, "{}_ratios_avg_change_in_reu_risk_neutral.csv".format(results_str))

    # risk averse
    risk_coeff = dmraev_calibration(a)
    reus_all_actions_a, changes_in_reu_a = get_reus_all_actions(risk_coeff, daly_burden_by_harm, states_dict, daly_burdens_dict, joint_probs_dict, event_probs, \
                                                            amf_dalys_per_1000, hens_sc_dalys_per_1000, shrimp_slaughter_sc_dalys_per_1000, shrimp_nh3_sc_dalys_per_1000, \
                                                            conservative_xrisk_reduction_per_bn, risky_xrisk_reduction_per_bn, to_print)
    reu_df_a = make_summary_stats_reu_df(reus_all_actions_a, risk_coeff, "{}_REU_risk_averse.csv".format(results_str))
    changes_in_reu_df_a = make_summary_stats_reu_df(changes_in_reu_a, risk_coeff, "{}_changes_in_REU_risk_averse.csv".format(results_str))
    prop_time_best_action_a, df_prop_time_best_action_a = get_proportion_time_best_action(reus_all_actions_a, "{}_variance_in_best_action_risk_averse.csv".format(results_str))
    df_ratio_of_changes_in_reu_a = get_ratio_of_change_in_reu_by_cause(changes_in_reu_a, "{}_ratios_avg_change_in_reu_risk_averse.csv".format(results_str))
    
    if to_print:
        print("Risk neutral REUs")
        print(reu_df_1)
        print("Risk Neutral changes in REUs")
        print(changes_in_reu_df_1)
        print("Risk Neutral proportion of time best action")
        print(df_prop_time_best_action_1)
        print("Risk neutral ambiguity-averse expected values")
        print("Risk averse, a={} REUs".format(a))
        print(reu_df_a)
        print("Risk averse, a={} changes in REUs".format(a))
        print(changes_in_reu_df_a)
        print("Risk averse, a={} proportion of time best action".format(a))
        print(df_prop_time_best_action_a)


    return reu_df_1, changes_in_reu_df_1, df_prop_time_best_action_1, df_ratio_of_changes_in_reu_1, \
        reu_df_a, changes_in_reu_df_a, df_prop_time_best_action_a, df_ratio_of_changes_in_reu_a

