### Existential an non-existential risk mitigation CEA model

import numpy as np
import pandas as pd
import squigglepy as sq
from squigglepy.numbers import K, M, B
import copy
import math 

N = 8*M

bp_to_decimal = lambda x: x/10000


def get_non_xrisk_probability(intervention_dict, extent_of_effect, years, to_print=False):
    '''
    Find the probability of non-existential risk events for the duration of the period of impact, up to 200 years. 
        These events are of magnitudes as follows: 3M to 30M, 30M to 300M, 300M to 3B. I also randomly sample the 
        reduction of the effect, and calculate the decrease in the cumulative probability of each sized event. 
    '''
    low_years_between_events_killing_3M = intervention_dict['low_years_between_events_killing_3M']
    high_years_between_events_killing_3M = intervention_dict['high_years_between_events_killing_3M']

    cost = intervention_dict['cost']

    years_between_events_killing_3M = sq.sample(sq.lognorm(low_years_between_events_killing_3M, high_years_between_events_killing_3M), N)

    power = sq.sample(sq.uniform(0.4, 0.7),N)

    years[years >200] = 200

    # probability of an event 10x as large scales with (1/10)**power 
    p_3M_to_30M = (1/years_between_events_killing_3M)*(1-(1/10)**power)
    p_30M_to_300M = (1/years_between_events_killing_3M)*(1/10)**power*(1-(1/10)**power)
    p_300M_to_3B = (1/years_between_events_killing_3M)*(1/10)**(2*power)*(1-(1/10)**power)

    cumulative_p_3M_to_30M = np.ones(N) - (np.ones(N) - p_3M_to_30M)**years
    cumulative_p_30M_to_300M = np.ones(N) - (np.ones(N) - p_30M_to_300M)**years
    cumulative_p_300M_to_3B = np.ones(N) - (np.ones(N) - p_300M_to_3B)**years

    relative_risk_reduction_3M_to_30M = sq.sample(sq.lognorm(0.01, 0.05), N)*extent_of_effect*cost/(1*B)
    relative_risk_reduction_30M_to_300M = sq.sample(sq.lognorm(0.01, 0.05), N)*extent_of_effect*cost/(1*B)
    relative_risk_reduction_300M_to_3B = sq.sample(sq.lognorm(0.001, 0.01), N)*extent_of_effect*cost/(1*B)

    post_int_p_3M_to_30M = (np.ones(N)-relative_risk_reduction_3M_to_30M)*p_3M_to_30M
    post_int_p_30M_to_300M = (np.ones(N)-relative_risk_reduction_30M_to_300M)*p_30M_to_300M
    post_int_p_300M_to_3B = (np.ones(N)-relative_risk_reduction_300M_to_3B)*p_300M_to_3B

    cumulative_post_int_p_3M_to_30M = np.ones(N) - (np.ones(N) - post_int_p_3M_to_30M)**years
    cumulative_post_int_p_30M_to_300M = np.ones(N) - (np.ones(N) - post_int_p_30M_to_300M)**years
    cumulative_post_int_p_300M_to_3B = np.ones(N) - (np.ones(N) - post_int_p_300M_to_3B)**years

    decrease_cumulative_p_3M_to_30M = cumulative_p_3M_to_30M - cumulative_post_int_p_3M_to_30M
    decrease_cumulative_p_30M_to_300M = cumulative_p_30M_to_300M - cumulative_post_int_p_30M_to_300M
    decrease_cumulative_p_300M_to_3B = cumulative_p_300M_to_3B - cumulative_post_int_p_300M_to_3B

    if to_print:
        print("Mean effect multiplier: {}".format(np.mean(extent_of_effect)))
        print("Percentiles: ")
        print(sq.get_percentiles(extent_of_effect))

        print("Mean years credit: {}".format(np.mean(years)))
        print("Percentiles: ")
        print(sq.get_percentiles(years))


        print("3M to 30M")
        print("Mean pre-intervention annual probability of event: {}".format(np.mean(p_3M_to_30M)))
        print("Mean pre-intervention cumulative probability of event: {}".format(np.mean(cumulative_p_3M_to_30M)))
        print("Mean post-intervention annual probability of event: {}".format(np.mean(post_int_p_3M_to_30M)))
        print("Mean post-intervention cumulative probability of event: {}".format(np.mean(cumulative_post_int_p_3M_to_30M)))
        print("Mean decrease in cumulative probability of event: {}".format(np.mean(decrease_cumulative_p_3M_to_30M)))
        print("Percentiles: ")
        print(sq.get_percentiles(decrease_cumulative_p_3M_to_30M))

        print("30M to 300M")
        print("Mean pre-intervention annual probability of event: {}".format(np.mean(p_30M_to_300M)))
        print("Mean pre-intervention cumulative probability of event: {}".format(np.mean(cumulative_p_30M_to_300M)))
        print("Mean post-intervention annual probability of event: {}".format(np.mean(post_int_p_30M_to_300M)))
        print("Mean post-intervention cumulative probability of event: {}".format(np.mean(cumulative_post_int_p_30M_to_300M)))
        print("Mean decrease in cumulative probability of event: {}".format(np.mean(decrease_cumulative_p_30M_to_300M)))
        print("Percentiles: ")
        print(sq.get_percentiles(decrease_cumulative_p_30M_to_300M))

        print("300M to 3B")
        print("Mean pre-intervention annual probability of event: {}".format(np.mean(p_300M_to_3B)))
        print("Mean pre-intervention cumulative probability of event: {}".format(np.mean(cumulative_p_300M_to_3B)))
        print("Mean post-intervention annual probability of event: {}".format(np.mean(post_int_p_300M_to_3B)))
        print("Mean post-intervention cumulative probability of event: {}".format(np.mean(cumulative_post_int_p_300M_to_3B)))
        print("Mean decrease in cumulative probability of event: {}".format(np.mean(decrease_cumulative_p_300M_to_3B)))
        print("Percentiles: ")
        print(sq.get_percentiles(decrease_cumulative_p_300M_to_3B))

    return decrease_cumulative_p_3M_to_30M, decrease_cumulative_p_30M_to_300M, decrease_cumulative_p_300M_to_3B

def get_amount_risk_can_eliminate_by_money_spent_model(intervention_dict):
    
    low_amt_risk_eliminated_per_B = intervention_dict['low_amt_risk_reduced_per_B']
    high_amt_risk_eliminated_per_B = intervention_dict['high_amt_risk_reduced_per_B']

    cost = intervention_dict['cost']

    low_amt_risk_eliminated_by_money_spent = low_amt_risk_eliminated_per_B*cost/(1*B)
    high_amt_risk_eliminated_by_money_spent = high_amt_risk_eliminated_per_B*cost/(1*B)

    model_risk_eliminated_if_good_by_money_spent = sq.lognorm(low_amt_risk_eliminated_by_money_spent, \
                                                              high_amt_risk_eliminated_by_money_spent, \
                                                     lclip = 0.1*low_amt_risk_eliminated_by_money_spent, rclip = 1)

    return model_risk_eliminated_if_good_by_money_spent

def get_dalys_saved(intervention_dict, to_print=False):
    '''
    Based on parameters specified in the intervention dictionary, return a vector of variables representing
        whether the intervention had an effect on xrisk, the effect size, whether the effect prevented or caused an x-risk,
        and the number of DALYs saved for an intervention. It also runs the non-existential risk probability reduction, 
        and calculates the DALYs saved from non-xrisk events (if any). 

    I also get a vector of expected values saved by the intervention for the risk attitudes 
        where what we care about is changing the probability of xrisks, and an estimate for 
        the DALY burden of extinction and non-existential catastrophes for the REU function. 
    '''
    p_has_effect = intervention_dict['p_has_effect']
    p_good_if_effect = intervention_dict['p_good_if_effect']

    low_extent_of_negative_effect = intervention_dict['low_extent_of_negative_effect']
    high_extent_of_negative_effect = intervention_dict['high_extent_of_negative_effect']

    # model if there was a good effect, how much risk was reduced per Amount of Money Spent. 
    model_risk_eliminated_if_good_by_money_spent = get_amount_risk_can_eliminate_by_money_spent_model(intervention_dict)

    model_extent_of_bad_effect = sq.lognorm(low_extent_of_negative_effect, high_extent_of_negative_effect, lclip = 0.05, rclip = 1)
    value_of_future, years_credit = get_value_of_future(intervention_dict)

    lives_lost_3M_to_30M_if_event = sq.sample(sq.lognorm(3*M, 30*M, lclip=3*M, rclip=30*M), N)
    lives_lost_30M_to_300M_if_event = sq.sample(sq.lognorm(30*M, 300*M, lclip=30*M, rclip=300*M), N)
    lives_lost_300M_to_3B_if_event = sq.sample(sq.lognorm(300*M, 3000*M, lclip=300*M, rclip=3000*M), N)

    # make default zero vectors for direction of effect, size of effect, whether it makes a difference, and DALYs saved
    effect_good_bad_zero = np.zeros(N)
    extent_of_effect = np.zeros(N)

    # generate whether effect was positive, negative, or zero, and the extent of the effect 
    for i in range(N):
        has_effect_i = sq.sample(sq.bernoulli(p_has_effect), 1)
        if has_effect_i == 1:
            is_good_value = sq.sample(sq.uniform(0,1), 1)
            if is_good_value < p_good_if_effect:
                # direction is positive
                effect_good_bad_zero[i] = 1
                extent_of_effect[i] = 1
            else:
                # direction is negative
                effect_good_bad_zero[i] = -1
                extent_neg_effect_i = sq.sample(model_extent_of_bad_effect, 1)
                extent_of_effect[i] = -1*extent_neg_effect_i

    effect_size = np.zeros(N)
    makes_difference = np.zeros(N)

    makes_difference_3M_to_30M = np.zeros(N)
    makes_difference_30M_to_300M = np.zeros(N)
    makes_difference_300M_to_3B = np.zeros(N)

    # non- xrisk DALYs saved
    lives_saved_non_xrisk = np.zeros(N)
    dalys_per_life = sq.sample(sq.uniform(32, 52), N)

    decrease_cumulative_p_3M_to_30M, decrease_cumulative_p_30M_to_300M, decrease_cumulative_p_300M_to_3B = \
        get_non_xrisk_probability(intervention_dict, extent_of_effect, years_credit, to_print)

    dalys_saved = np.zeros(N) 

    expected_dalys_saved = np.zeros(N)

    for i in range(N):
        decrease_cumulative_p_3M_to_30M_i = decrease_cumulative_p_3M_to_30M[i]
        decrease_cumulative_p_30M_to_300M_i = decrease_cumulative_p_30M_to_300M[i]
        decrease_cumulative_p_300M_to_3B_i = decrease_cumulative_p_300M_to_3B[i]

        effect_good_bad_zero_i = effect_good_bad_zero[i]
        effect_size_if_good_i = sq.sample(model_risk_eliminated_if_good_by_money_spent, 1)
        if effect_good_bad_zero_i == 1:
            # find reduction in risk and store
            effect_size_i = effect_size_if_good_i
            effect_size[i] = effect_size_i

            # find expected dalys saved due to preventing xrisks
            expected_dalys_saved[i] = value_of_future[i]*effect_size_i

            # determine if makes a difference and value saved
            made_difference_i = sq.sample(sq.bernoulli(abs(effect_size_i)), 1)
            if made_difference_i == 1:
                makes_difference[i] = made_difference_i
                dalys_saved_i = value_of_future[i]
                dalys_saved[i] = dalys_saved_i

            # non-xrisk dalys
            makes_difference_3M_to_30M_i = sq.sample(sq.bernoulli(decrease_cumulative_p_3M_to_30M_i), 1)
            makes_difference_30M_to_300M_i = sq.sample(sq.bernoulli(decrease_cumulative_p_30M_to_300M_i), 1)
            makes_difference_300M_to_3B_i = sq.sample(sq.bernoulli(decrease_cumulative_p_300M_to_3B_i), 1)

            makes_difference_3M_to_30M[i] = makes_difference_3M_to_30M_i
            makes_difference_30M_to_300M[i] = makes_difference_30M_to_300M_i
            makes_difference_300M_to_3B[i] = makes_difference_300M_to_3B_i

            lives_saved_non_xrisk_i = 0
            expected_lives_saved_non_xrisk_i = 0

            # lives saved by averting 3M-30M catastrophes
            lives_saved_non_xrisk_i += makes_difference_3M_to_30M_i*lives_lost_3M_to_30M_if_event[i]
            expected_lives_saved_non_xrisk_i += decrease_cumulative_p_3M_to_30M_i*lives_lost_3M_to_30M_if_event[i]

            # lives saved by averting 30-300M catastrophes
            lives_saved_non_xrisk_i += makes_difference_30M_to_300M_i*lives_lost_30M_to_300M_if_event[i]
            expected_lives_saved_non_xrisk_i += decrease_cumulative_p_30M_to_300M_i*lives_lost_30M_to_300M_if_event[i]

            # lives saved by averting 300M-3B catastrophes
            lives_saved_non_xrisk_i += makes_difference_300M_to_3B_i*lives_lost_300M_to_3B_if_event[i]
            expected_lives_saved_non_xrisk_i += decrease_cumulative_p_300M_to_3B_i*lives_lost_300M_to_3B_if_event[i]

            lives_saved_non_xrisk[i] = lives_saved_non_xrisk_i

            dalys_saved[i] += lives_saved_non_xrisk_i*dalys_per_life[i]
            expected_dalys_saved[i] += expected_lives_saved_non_xrisk_i*dalys_per_life[i]
                
        elif effect_good_bad_zero_i == -1:
            # find reduction in risk and store
            extent_neg_effect_i = extent_of_effect[i]
            effect_size_i = extent_neg_effect_i*effect_size_if_good_i
            effect_size[i] = effect_size_i

            expected_dalys_saved[i] = value_of_future[i]*effect_size_i
            # determine if makes a difference
            made_difference_i = sq.sample(sq.bernoulli(abs(effect_size_i)), 1)

            if made_difference_i == 1:
                makes_difference[i] = -1*made_difference_i
                dalys_saved_i = -1*value_of_future[i]
                dalys_saved[i] = dalys_saved_i 

            # non-xrisk dalys
            makes_difference_3M_to_30M_i = -1*sq.sample(sq.bernoulli(abs(decrease_cumulative_p_3M_to_30M_i)), 1)
            makes_difference_30M_to_300M_i = -1*sq.sample(sq.bernoulli(abs(decrease_cumulative_p_30M_to_300M_i)), 1)
            makes_difference_300M_to_3B_i = -1*sq.sample(sq.bernoulli(abs(decrease_cumulative_p_300M_to_3B_i)), 1)

            makes_difference_3M_to_30M[i] = makes_difference_3M_to_30M_i
            makes_difference_30M_to_300M[i] = makes_difference_30M_to_300M_i
            makes_difference_300M_to_3B[i] = makes_difference_300M_to_3B_i

            lives_saved_non_xrisk_i = 0
            expected_lives_saved_non_xrisk_i = 0

            lives_saved_non_xrisk_i += makes_difference_3M_to_30M_i*lives_lost_3M_to_30M_if_event[i]
            expected_lives_saved_non_xrisk_i += decrease_cumulative_p_3M_to_30M_i*lives_lost_3M_to_30M_if_event[i]

            lives_saved_non_xrisk_i += makes_difference_30M_to_300M_i*lives_lost_30M_to_300M_if_event[i]
            expected_lives_saved_non_xrisk_i += decrease_cumulative_p_30M_to_300M_i*lives_lost_30M_to_300M_if_event[i]

            lives_saved_non_xrisk_i += makes_difference_300M_to_3B_i*lives_lost_300M_to_3B_if_event[i]
            expected_lives_saved_non_xrisk_i += decrease_cumulative_p_300M_to_3B_i*lives_lost_300M_to_3B_if_event[i]

            lives_saved_non_xrisk[i] = lives_saved_non_xrisk_i
            dalys_saved[i] += lives_saved_non_xrisk_i*dalys_per_life[i]
            expected_dalys_saved[i] += expected_lives_saved_non_xrisk_i*dalys_per_life[i]

        else:
            if to_print and i == 1*M:
                print("No effect")
                print(effect_size[i])
                print(extent_of_effect[i])
                print(effect_good_bad_zero[i])
                print(makes_difference[i])
                print(dalys_saved[i])
                print(expected_dalys_saved[i])
                print(lives_saved_non_xrisk[i])
                print(decrease_cumulative_p_3M_to_30M[i])
                print(decrease_cumulative_p_30M_to_300M[i])
                print(decrease_cumulative_p_300M_to_3B[i])  
                print("Should all be zero")

    dalys_saved_approx_if_prevent_risk = np.mean(lives_saved_non_xrisk[lives_saved_non_xrisk > 0])*dalys_per_life + value_of_future

    if to_print:
        print("Number 3M to 30M where difference made: {}".format(np.sum(abs(makes_difference_3M_to_30M))))
        print("Number 30M to 300M where difference made: {}".format(np.sum(abs(makes_difference_30M_to_300M))))
        print("Number 300M to 3B where difference made: {}".format(np.sum(abs(makes_difference_300M_to_3B))))

        print("Total 3M to 30M prevented {}".format(np.sum(makes_difference_3M_to_30M[makes_difference_3M_to_30M == 1])))
        print("Total 30M to 300M prevented {}".format(np.sum(makes_difference_30M_to_300M[makes_difference_30M_to_300M == 1])))
        print("Total 300M to 3B prevented {}".format(np.sum(makes_difference_300M_to_3B[makes_difference_300M_to_3B == 1])))

        print("Total 3M to 30M caused {}".format(abs(np.sum(makes_difference_3M_to_30M[makes_difference_3M_to_30M == -1]))))
        print("Total 30M to 300M caused {}".format(abs(np.sum(makes_difference_30M_to_300M[makes_difference_30M_to_300M == -1]))))
        print("Total 300M to 3B caused {}".format(abs(np.sum(makes_difference_300M_to_3B[makes_difference_300M_to_3B == -1]))))

        print("Mean lives saved non-xrisk: {}".format(np.mean(lives_saved_non_xrisk)))
        print("Percentiles: {}".format(sq.get_percentiles(lives_saved_non_xrisk)))

        print("Mean DALYs saved approx if prevent risk: {}".format(np.mean(dalys_saved_approx_if_prevent_risk)))
        print("Expected DALYs saved: {}".format(np.mean(expected_dalys_saved)))
        print("Percentiles Expected DALYs saved: {}".format(sq.get_percentiles(expected_dalys_saved)))


    return effect_good_bad_zero, effect_size, makes_difference, dalys_saved, dalys_saved_approx_if_prevent_risk, expected_dalys_saved


def get_value_of_future(intervention_dict):
    '''
    The counterfactual value saved by eliminating xrisk. 
    '''
    low_years_credit = intervention_dict['low_years_credit']
    high_years_credit = intervention_dict['high_years_credit']

    years_credit = sq.sample(sq.norm(low_years_credit, high_years_credit, lclip=low_years_credit*0.1, rclip=high_years_credit*10), N)

    avg_dalys_per_year = sq.sample(sq.uniform(0.7, 1.0), N)
    avg_number_people = sq.sample(sq.lognorm(7*B, 12*B, lclip=6*B, rclip=15*B), N)

    value_of_future = years_credit*avg_dalys_per_year*avg_number_people

    return value_of_future, years_credit


def get_dalys_saved_per_1000(intervention_dict, to_print=False):
    '''
    Get distribution of DALYs saved per $1000 spent on intervention, and expected DALYs saved per $1000. 

    Also generate statistics on the frequency that the intervention makes a difference, 
        both positive, negative, or both, and mean effect size. 
    '''

    effect_good_bad_zero, effect_size, makes_difference, dalys_saved, dalys_saved_approx_prevent_xrisk, expected_dalys_saved  = get_dalys_saved(intervention_dict, to_print)

    count_nonzero = np.sum(abs(makes_difference))
    p_neg = np.sum(makes_difference < 0)/N
    p_pos = np.sum(makes_difference > 0)/N
    p_difference = count_nonzero/N

    mean_effect_size = np.mean(effect_size)

    cost = intervention_dict['cost']

    dalys_saved_per_1000 = dalys_saved*1000/cost

    expected_dalys_saved_per_1000 = expected_dalys_saved*1000/cost

    if to_print:
        print("Prop. time intervention affects risk: {}".format(np.mean(abs(effect_good_bad_zero))))
        print("Mean effect size (per ${} spent): {}".format(cost, np.mean(effect_size)))
        print("Mean absolute effect size: {}".format(np.mean(abs(effect_size))))
        print("Approx number makes a difference: {}".format(N*np.mean(abs(effect_size))))
        print("Obeserved number makes a difference: {}".format(count_nonzero))
        print("Chance prevents extinction: {}%".format(np.round(100*p_pos,8)))
        print("Chance causes extinction: {}%".format(np.round(100*p_neg, 8)))
        print("Chance make a difference: {}%".format(np.round(100*p_difference, 8)))
        print("AVG DALYs saved (per ${} spent): ".format(cost, np.mean(dalys_saved)))
        print("Avg. DALYs saved per $1000: {}".format(np.mean(dalys_saved_per_1000)))
        print("Percentiles: ")
        print(np.percentile(np.round(dalys_saved_per_1000,2), [0.0001, 0.001, 0.01, 0.1, 1, 5, 50, 95, 99, 99.9, 99.99, 99.999]))
        print("Expected DALYs saved per $1000: {}".format(np.mean(expected_dalys_saved_per_1000)))
        print("Percentiles: ")
        print(np.percentile(np.round(expected_dalys_saved_per_1000,2), [0.0001, 0.001, 0.01, 0.1, 1, 5, 50, 95, 99, 99.9, 99.99, 99.999]))


    return dalys_saved_per_1000, effect_size, mean_effect_size, p_pos, p_neg, dalys_saved_approx_prevent_xrisk, expected_dalys_saved_per_1000
