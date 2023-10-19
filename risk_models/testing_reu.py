import reu as reu
import numpy as np


sq = {'dead': {'daly burden': [-100], 'joint probability': [0.1]}, 
      'normal': {'daly burden': [0], 'joint probability':[ 0.8]},
      'utopia': {'daly burden': [100], 'joint probability': [0.1]}}

predicted_reu_sq = 0.8*0 + 0.1*100 + 0.1*(-100)
print(predicted_reu_sq)


sip = {'dead': {'daly burden': [-100], 'joint probability': [0.1]},
        'normal': {'daly burden': [1], 'joint probability': [0.8]},
        'utopia': {'daly burden': [100], 'joint probability': [0.1]}}

predicted_reu_sip = 0.8*1 + 0.1*100 + 0.1*(-100)
print(predicted_reu_sip)

ei = {'dead': {'daly burden': [-100], 'joint probability': [0.09]},
        'normal': {'daly burden': [0], 'joint probability': [0.81]},
        'utopia': {'daly burden': [100], 'joint probability': [0.1]}}

predicted_reu_ei = 0.81*0 + 0.09*(-100) + 0.1*100
print(predicted_reu_ei)

el = {'dead': {'daly burden': [-100], 'joint probability': [0.1]},
        'normal': {'daly burden': [0], 'joint probability': [0.79]},
        'utopia': {'daly burden': [100], 'joint probability':[0.11]}}

predicted_reu_el = 0.79*0 + 0.1*(-100) + 0.11*(100)
print(predicted_reu_el)

lotteries = {'sq': sq, 'sip': sip, 'ei': ei, 'el': el}

def test_states_ranking(l): 
    ranked_states = []
    for lottery_name, lottery in l.items():
        ranked_state_utility_probs_l = reu.get_ranked_state_tuples_one_sim(0, lottery)
        ranked_states.append(ranked_state_utility_probs_l)
    return ranked_states

def test_get_prob_at_least_as_good(l):
    ranked_states = test_states_ranking(l)
    ranked_cumulative_probs = []
    for ranked_state_utility_probs_l in ranked_states:
        ranked_cumulative_probs_l = reu.get_probability_of_at_least_as_good_one_sim(ranked_state_utility_probs_l)
        ranked_cumulative_probs.append(ranked_cumulative_probs_l)
    return ranked_cumulative_probs

def test_get_reu_one_sim(l):
    ranked_cumulative_probs = test_get_prob_at_least_as_good(l)
    count = 0
    reus = {}
    for lottery_name, lottery in l.items():
        reus[lottery_name] = 0
        reu_one_sim = reu.get_reu_one_sim(ranked_cumulative_probs[count], 1)
        reus[lottery_name] = reu_one_sim
        count += 1
    return reus

ranked_states = test_states_ranking(lotteries)
print("Ranked States: ")
print(ranked_states)

cumulative_probs = test_get_prob_at_least_as_good(lotteries)
print("Cumulative Probs: ")
print(cumulative_probs)

reus = test_get_reu_one_sim(lotteries)
print("REUs: ")
print(reus)

