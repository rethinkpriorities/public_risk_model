### Road Safety as a risky GHD Intervention based on an RP report on the subject

import numpy as np
import squigglepy as sq
from squigglepy.numbers import K, M, B

## Most traffic deaths are amongst adults, 
##  so we use 32 DALYs averted per life saved from traffic fatality
DALYS_PER_LIFE = 32
N = 6*M


## Dictionaries containing the distributions for variables considered in the CEA model for each law 
##    passed in China (a DUI law) and Vietnam (a DUI law and a helmet law). See the RP report for details. 
china_dui_dict = {'direct_speed_up': sq.uniform(0,0), 
              'prop_philanthropy_credit': sq.uniform(0, 0.15),
              'counterfactual_time_to_get_equal_law': sq.lognorm(3, 8),
              'prop_fatalities_averted_by_law': 0.04,
              'actual_effect_size': sq.norm(0.4, 0.8, lclip=0.2, rclip=1),
              'total_fatalities_period_impact': 1717617,
              'years': 6}

vietnam_helmet_dict = {'direct_speed_up': sq.lognorm(0.15, 0.38),
                'prop_philanthropy_credit': sq.uniform(0.7, 0.9), 
                'counterfactual_time_to_get_equal_law': sq.lognorm(5, 17),
                'prop_fatalities_averted_by_law': 0.06,
                'actual_effect_size': sq.norm(0.6, 1, lclip=0.6, rclip=1),
                'total_fatalities_period_impact': 275206,
                'years': 11}

vietnam_dui_dict = {'direct_speed_up': sq.lognorm(0.15, 0.38),
                'prop_philanthropy_credit': sq.uniform(0.1, 0.3), 
                'counterfactual_time_to_get_equal_law': sq.lognorm(2, 7),
                'prop_fatalities_averted_by_law': 0.13,
                'actual_effect_size': sq.norm(0.2, 0.6, lclip=0, rclip=0.8),
                'total_fatalities_period_impact': 275206,
                'years': 11}

def get_speed_up_of_law(law_dict, to_print=False):
    '''
    Estimate the years by which philanthropic spending on the road safety legislation campaign 
        sped up its passage and effective implementation. 
    These estimates are returned as a vector of simulations. 
    '''
    direct_speed_up_model = law_dict['direct_speed_up']
    prop_philanthropy_credit_model = law_dict['prop_philanthropy_credit']
    counterfactual_time_to_get_equal_law_model = law_dict['counterfactual_time_to_get_equal_law']

    direct_speed_up = sq.sample(direct_speed_up_model, N)
    prop_philanthropy_credit = sq.sample(prop_philanthropy_credit_model, N)
    counterfactual_time_to_get_equal_law = sq.sample(counterfactual_time_to_get_equal_law_model, N)

    speed_up_of_law = direct_speed_up + prop_philanthropy_credit * counterfactual_time_to_get_equal_law

    if to_print:
        print('Direct speed up: ', np.mean(direct_speed_up))
        print('Prop philanthropy credit: ', np.mean(prop_philanthropy_credit))
        print('Counterfactual time to get equal law: ', np.mean(counterfactual_time_to_get_equal_law))
        print('Speed up of law: ', np.mean(speed_up_of_law))
        print('Percentiles: ', np.percentile(speed_up_of_law, [5, 50, 95]))

    return speed_up_of_law

def get_adjusted_effect_of_law(dict, to_print=False):
    '''
    Returns a vector of the estimated true effect of the law (% decline in fatalities). The report discounts the 
        estimated effect from studies by some estimated amount. 
    '''
    prop_fatalities_averted_by_law = dict['prop_fatalities_averted_by_law']
    actual_effect_size_model = dict['actual_effect_size']

    actual_effect_size = sq.sample(actual_effect_size_model, N)
    adjusted_effect_of_law = prop_fatalities_averted_by_law * actual_effect_size

    if to_print:
        print('Prop fatalities averted by law: ', prop_fatalities_averted_by_law)
        print('Prop actual effect size: ', np.mean(actual_effect_size))
        print('Adjusted effect of law: ', np.mean(adjusted_effect_of_law))
        print('Percentiles: ', np.percentile(adjusted_effect_of_law, [5, 50, 95]))

    return adjusted_effect_of_law

def get_lives_saved_by_philanthropy(dict, to_print=False):
    '''
    Take the total lives saved per year of counterfactual impact, the length of impact, 
        and the estimated decrease in deaths per year. Return a vector of the number lives 
        saved counterfactually by philanthropic spending on the laws. 
    '''
    total_fatalities_period_impact = dict['total_fatalities_period_impact']
    years = dict['years']

    avg_fatalities_per_year = total_fatalities_period_impact / years
    
    adjusted_effect_of_law = get_adjusted_effect_of_law(dict)

    # total lives saved per year by the law (not by philanthropy)
    lives_saved_by_law_per_yr = avg_fatalities_per_year * adjusted_effect_of_law

    # philanthropy credit
    speed_up_of_law = get_speed_up_of_law(dict)
    lives_saved_by_philanthropy = speed_up_of_law * lives_saved_by_law_per_yr

    if to_print:
        print('Lives saved by law per year: ', np.mean(lives_saved_by_law_per_yr))
        print('Lives saved by philanthropy: ', np.mean(lives_saved_by_philanthropy))
        print('Percentiles: ', np.percentile(lives_saved_by_philanthropy, [5, 50, 95]))

    return lives_saved_by_philanthropy

def get_dalys_saved_by_philanthropy(dict, to_print = False):
    '''
    Translate the number of lives saved by the law into DALYs averted. 
    '''
    lives_saved_by_philanthropy = get_lives_saved_by_philanthropy(dict)
    dalys_saved_by_philanthropy = lives_saved_by_philanthropy * DALYS_PER_LIFE

    if to_print:
        print('DALYs saved by philanthropy: ', np.mean(dalys_saved_by_philanthropy))
        print('Percentiles: ', np.percentile(dalys_saved_by_philanthropy, [5, 50, 95]))

    return dalys_saved_by_philanthropy

def vietnam_total_dalys_saved(to_print=False):
    '''
    Return a vector of the total DALYs saved by road safety philanthropy in Vietnam.
        This combines both the lives saved from the DUI law and the helmet law.
    '''
    dalys_saved_helmet_law = get_dalys_saved_by_philanthropy(vietnam_helmet_dict)
    dalys_saved_dui_law = get_dalys_saved_by_philanthropy(vietnam_dui_dict)

    vietnam_total_dalys_saved = dalys_saved_helmet_law + dalys_saved_dui_law

    if to_print: 
        print('Vietnam total DALYs saved: ', np.mean(vietnam_total_dalys_saved))
        print('Percentiles: ', np.percentile(vietnam_total_dalys_saved, [5, 50, 95]))
    
    return vietnam_total_dalys_saved

def get_overall_dalys_per_1000(to_print=False):
    '''
    Given the costs spent by the philanthropists on each campaign in China
        and Vietnam, calculate the number of DALYs averted per $1000 spent in total. 
    This is not yet adjusted for the probability of future laws passing in other countries. 
    '''
    china_dalys_saved_by_philanthropy = get_dalys_saved_by_philanthropy(china_dui_dict)
    china_cost = sq.sample(sq.uniform(600*K, 1400*K), N)

    vietnam_dalys_saved = vietnam_total_dalys_saved()
    vietnam_cost_helmet = sq.sample(sq.uniform(1.5*M, 3.9*M), N)
    vietnam_cost_dui = sq.sample(sq.uniform(1.75*M, 2.75*M), N)
    vietnam_total_cost = vietnam_cost_helmet + vietnam_cost_dui

    overall_dalys_saved = china_dalys_saved_by_philanthropy + vietnam_dalys_saved
    overall_cost = china_cost + vietnam_total_cost

    overall_dalys_per_1000 = overall_dalys_saved / overall_cost * 1000

    if to_print:
        print('China DALYs saved by philanthropy: ', np.mean(china_dalys_saved_by_philanthropy))
        print('Vietnam DALYs saved: ', np.mean(vietnam_dalys_saved))
        print('China cost: ', np.mean(china_cost))
        print('Vietnam cost: ', np.mean(vietnam_total_cost))
        print('Overall DALYs saved: ', np.mean(overall_dalys_saved))
        print('Overall cost: ', np.mean(overall_cost))
        print('Overall cost-effectiveness: ', np.mean(overall_dalys_per_1000))
        print('Percentiles: ', np.percentile(overall_dalys_per_1000, [5, 50, 95]))

    return overall_dalys_per_1000

def get_failure_adjusted_dalys_per_1000(to_print=False): 
    '''
    Assuming the probability that future campaigns will be successful 
        at passing legislation is between 19% and 48%, I adjust the 
        cost-effectiveness of road safety laws accordingly. This is done by 
        randomly selecting a bernoulli variable to indicate whether the law passed
        or not, based on a randomly drawn probability of success. 
    This is the main function that's used in the cost-effectiveness estimates. 
    '''
    unadjusted_success_rate = 0.48
    adjustment_factor = sq.sample(sq.uniform(0.3, 0.7), N)

    risk_weighted_success_rate = unadjusted_success_rate * adjustment_factor

    did_succeed = np.zeros(N)

    for i in range(N):
        X = sq.sample(sq.binomial(1, risk_weighted_success_rate[i]), 1)
        did_succeed[i] = X

    unadjusted_cost_effectiveness = get_overall_dalys_per_1000()

    risk_weighted_cost_effectiveness = unadjusted_cost_effectiveness * did_succeed

    if to_print:
        print('Unadjusted success rate: ', unadjusted_success_rate)
        print('Adjustment factor: ', np.mean(adjustment_factor))
        print('Risk-weighted success rate: ', np.mean(risk_weighted_success_rate))
        print('Unadjusted cost-effectiveness: ', np.mean(unadjusted_cost_effectiveness))
        print('Risk-weighted cost-effectiveness: ', np.mean(risk_weighted_cost_effectiveness))
        print('Percentiles: ', np.percentile(risk_weighted_cost_effectiveness, [5, 50, 95]))
    
    return risk_weighted_cost_effectiveness
