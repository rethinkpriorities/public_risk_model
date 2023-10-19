## testing the cause-comparison-specific REU functions
import numpy as np
import reu 
import squigglepy as sq
from squigglepy.numbers import K, M, B

interventions = ["Nothing", "AMF", "CF campaign", "Shrimp welfare - stunning", "Shrimp welfare - NH3", "Conservative x-risk work", "Risky x-risk work"]

## you have to change N in REU to 4 and delete the "risk_models" part in importing ambiguity aversion
N = 4

MONEY = 100*M

MULTIPLES_OF_1K = MONEY/1000

MULTIPLES_OF_1B = MONEY/(1*B)

# inputs
xrisk_dalys_at_stake = np.array([1000*B, 2000*B, 3000*B, 4000*B])
shrimp_slaughter_human_daly_burden = np.array([10*B, 20*B, 30*B, 40*B])
shrimp_nh3_human_daly_burden = np.array([10*B, 20*B, 30*B, 40*B])
malaria_daly_burden = np.array([10*B, 20*B, 30*B, 40*B])
hens_daly_burden = np.array([10*B, 20*B, 30*B, 40*B])

p_hens_sent = np.array([0.8, 0.9, 0.7, 0.8])
p_shrimp_sent = np.array([0.1, 0.2, 0.3, 0.4])
p_xrisk = np.array([0.1, 0.2, 0.3, 0.4])

amf_dalys_per_1000 = np.array([1,2,4,3])
hens_dalys_per_1000 = np.array([2000,4000,6000,8000])
shrimp_slaughter_dalys_per_1000 = np.array([1, 2, 3, 4])
shrimp_nh3_dalys_per_1000 = np.array([3,6,9,12])
conservative_xrisk_reduction_per_bn = np.array([-0.0001, 0.0005, -0.0002, 0.0008])
risky_xrisk_reduction_per_bn = np.array([-0.001, 0.005, -0.002, 0.008])

# define the expected values of the lotteries for "Nothing"
nothing_predicted_daly_burden_xcs = xrisk_dalys_at_stake+shrimp_slaughter_human_daly_burden+malaria_daly_burden+shrimp_nh3_human_daly_burden+hens_daly_burden
nothing_predicted_xcs_joint_prob = p_xrisk*p_shrimp_sent*p_hens_sent

nothing_predicted_daly_burden_xcccsc = malaria_daly_burden
nothing_predicted_xcccsc_joint_prob = (1-p_xrisk)*(1-p_shrimp_sent)*(1-p_hens_sent)

# define the expected values of the lotteries for "Conservative x-risk work"
cons_new_xrisk_prob = []
risky_new_xrisk_prob = []
for i in range(N):
    cons_new_xrisk_prob.append(max(p_xrisk[i] - conservative_xrisk_reduction_per_bn[i]*MULTIPLES_OF_1B, 0.0000001)) 
# define the expected values of the lotteries for "Risky x-risk work"
    risky_new_xrisk_prob.append(max(p_xrisk[i] - risky_xrisk_reduction_per_bn[i]*MULTIPLES_OF_1B, 0.0000001))

cons_new_xrisk_prob = np.array(cons_new_xrisk_prob)
risky_new_xrisk_prob = np.array(risky_new_xrisk_prob)

# define daly effects
amf_effect = np.array([min(malaria_daly_burden[i], amf_dalys_per_1000[i]*MULTIPLES_OF_1K) for i in range(N)])
hens_effect = np.array([min(hens_daly_burden[i], hens_dalys_per_1000[i]*MULTIPLES_OF_1K) for i in range(N)])
shrimp_slaughter_effect = np.array([min(shrimp_slaughter_human_daly_burden[i], shrimp_slaughter_dalys_per_1000[i]*MULTIPLES_OF_1K) for i in range(N)])
shrimp_nh3_effect = np.array([min(shrimp_nh3_human_daly_burden[i], shrimp_nh3_dalys_per_1000[i]*MULTIPLES_OF_1K) for i in range(N)])
new_joint_prob_conservative_xcs = reu.get_joint_prob_state(cons_new_xrisk_prob, p_shrimp_sent, p_hens_sent, 1, 1, 1)
new_joint_prob_conservative_xcccsc = reu.get_joint_prob_state(cons_new_xrisk_prob, p_shrimp_sent, p_hens_sent, 0, 0, 0)
new_joint_prob_risky_xcs = reu.get_joint_prob_state(risky_new_xrisk_prob, p_shrimp_sent, p_hens_sent, 1, 1, 1)
new_joint_prob_risky_xcccsc = reu.get_joint_prob_state(risky_new_xrisk_prob, p_shrimp_sent, p_hens_sent, 0, 0, 0)

predicted_lotteries_for_actions = {'Nothing': {'X/C/S': {'dalys': nothing_predicted_daly_burden_xcs, 
                                               'joint prob': nothing_predicted_xcs_joint_prob},
                                     'Xc/Cc/Sc': {'dalys': nothing_predicted_daly_burden_xcccsc, 
                                                  'joint prob': nothing_predicted_xcccsc_joint_prob}},
                        'AMF': {'X/C/S': {'dalys': nothing_predicted_daly_burden_xcs -  amf_effect, 
                                          'joint prob': nothing_predicted_xcs_joint_prob},
                                    'Xc/Cc/Sc': {'dalys': nothing_predicted_daly_burden_xcccsc - amf_effect, 
                                                 'joint prob': nothing_predicted_xcccsc_joint_prob}},
                        'CF campaign': {'X/C/S': {'dalys': nothing_predicted_daly_burden_xcs - hens_effect, 
                                                  'joint prob': nothing_predicted_xcs_joint_prob},
                                    'Xc/Cc/Sc': {'dalys': nothing_predicted_daly_burden_xcccsc, 
                                                 'joint prob': nothing_predicted_xcccsc_joint_prob}},
                        'Shrimp welfare - stunning': {'X/C/S': {'dalys': nothing_predicted_daly_burden_xcs - shrimp_slaughter_effect,
                                                                'joint prob': nothing_predicted_xcs_joint_prob},
                                    'Xc/Cc/Sc': {'dalys': nothing_predicted_daly_burden_xcccsc, 
                                                 'joint prob': nothing_predicted_xcccsc_joint_prob}},
                        'Shrimp welfare - NH3': {'X/C/S': {'dalys': nothing_predicted_daly_burden_xcs - shrimp_nh3_effect, 
                                                           'joint prob': nothing_predicted_xcs_joint_prob},
                                    'Xc/Cc/Sc': {'dalys': nothing_predicted_daly_burden_xcccsc, 
                                                 'joint prob': nothing_predicted_xcccsc_joint_prob}},
                        'Conservative x-risk work': {'X/C/S': {'dalys': nothing_predicted_daly_burden_xcs, 
                                                               'joint prob': new_joint_prob_conservative_xcs,},
                                    'Xc/Cc/Sc': {'dalys': nothing_predicted_daly_burden_xcccsc, 
                                                 'joint prob': new_joint_prob_conservative_xcccsc}},
                        'Risky x-risk work': {'X/C/S': {'dalys': nothing_predicted_daly_burden_xcs, 
                                                        'joint prob': new_joint_prob_risky_xcs},
                                    'Xc/Cc/Sc': {'dalys': nothing_predicted_daly_burden_xcccsc, 
                                                 'joint prob': new_joint_prob_risky_xcccsc}}
            }

def create_daly_burden_by_harm_dict():
    daly_burden_by_harm_dict = {'malaria': -1*malaria_daly_burden, #https://ourworldindata.org/burden-of-disease#the-disease-burden-by-cause
                    'x-risk': -1*xrisk_dalys_at_stake,
                    'chickens': -1*hens_daly_burden,
                    'shrimp - slaughter': -1*shrimp_slaughter_human_daly_burden,  
                    'shrimp - NH3': -1*shrimp_nh3_human_daly_burden}    
    return daly_burden_by_harm_dict

def create_event_probs_dict():
    event_probs = {'P(x-risk occurs)': p_xrisk,
               'P(chickens sentient)': p_hens_sent,
                'P(shrimp sentient)': p_shrimp_sent,}
    return event_probs
# make the joint probability dictionaries
def make_dictionaries_needed():
    daly_burden_by_harm_dict = create_daly_burden_by_harm_dict()
    event_probs = create_event_probs_dict()
    states_of_world_dict = reu.create_states_of_world_dict()
    joint_probs_dict = reu.make_joint_prob_states_dict(states_of_world_dict, event_probs)
    daly_burden_by_state_dict = reu.make_daly_burdens_dict(states_of_world_dict, daly_burden_by_harm_dict)
    return daly_burden_by_harm_dict, event_probs, states_of_world_dict, joint_probs_dict, daly_burden_by_state_dict

daly_burden_by_harm_dict, event_probs, states_of_world_dict, joint_probs_dict, daly_burden_by_state_dict = make_dictionaries_needed()

def get_one_lottery(action):
    lottery = reu.get_lottery_for_action(action, daly_burden_by_harm_dict, states_of_world_dict, daly_burden_by_state_dict, joint_probs_dict, \
                            event_probs, amf_dalys_per_1000, hens_dalys_per_1000, shrimp_slaughter_dalys_per_1000, \
                            shrimp_nh3_dalys_per_1000, conservative_xrisk_reduction_per_bn, risky_xrisk_reduction_per_bn, to_print=False)
    return lottery

def make_lotteries():
    lotteries = {}
    for action in interventions:
        lotteries[action] = get_one_lottery(action)
    return lotteries

def check_xcs_daly_burden(lottery, predicted_daly_burden_xcs):
    xcs_daly_burden = lottery['X/C/S']['daly burden']

    for i in range(N):
        if xcs_daly_burden[i] != -1*predicted_daly_burden_xcs[i]:
            print('The expected daly burden of the X/C/S intervention is incorrect.')
            print('Expected: ', -1*predicted_daly_burden_xcs[i])
            print('Actual: ', xcs_daly_burden[i])
            return
    print('The expected daly burden of the X/C/S intervention is correct.')
    print('Expected: ', -1*predicted_daly_burden_xcs)
    print('Actual: ', xcs_daly_burden)
    return

def check_xcs_joint_prob(lottery, predicted_xcs_joint_prob):
    xcs_joint_prob = lottery['X/C/S']['joint probability']

    for i in range(N):
        if round(xcs_joint_prob[i],6) != round(predicted_xcs_joint_prob[i],6):
            print('The joint probability of the X/C/S intervention is incorrect.')
            print('Expected: ', np.round(predicted_xcs_joint_prob[i],6))
            print('Actual: ', np.round(xcs_joint_prob[i],6))
            return
    print('The joint probability of the X/C/S intervention is correct.')
    print('Expected: ', np.round(predicted_xcs_joint_prob,6))
    print('Actual: ', np.round(xcs_joint_prob,6))

    return

def check_xcccsc_daly_burden(lottery, predicted_daly_burden_xcccsc):
    xcccsc_daly_burden = lottery['Xc/Cc/Sc']['daly burden']

    for i in range(N):
        if xcccsc_daly_burden[i] != -1*predicted_daly_burden_xcccsc[i]:
            print('The expected daly burden of the Xc/Cc/Sc intervention is incorrect.')
            print('Expected: ', -1*predicted_daly_burden_xcccsc[i])
            print('Actual: ', xcccsc_daly_burden[i])
            return

    print('The expected daly burden of the Xc/Cc/Sc intervention is correct.')
    print('Expected: ', -1*predicted_daly_burden_xcccsc)
    print('Actual: ', xcccsc_daly_burden)
    return

def check_xcccsc_joint_prob(lottery, predicted_xcccsc_joint_prob):
    xcccsc_joint_prob = lottery['Xc/Cc/Sc']['joint probability']

    for i in range(N):
        if round(xcccsc_joint_prob[i],6) != round(predicted_xcccsc_joint_prob[i],6):
            print('The joint probability of the Xc/Cc/Sc intervention is incorrect.')
            print('Expected: ', np.round(predicted_xcccsc_joint_prob[i],6))
            print('Actual: ', np.round(xcccsc_joint_prob[i],6))
            return

    print('The joint probability of the Xc/Cc/Sc intervention is correct.')
    print('Expected: ', np.round(predicted_xcccsc_joint_prob,6))
    print('Actual: ', np.round(xcccsc_joint_prob,6))
    return

def check_lottery(action):
    lottery = get_one_lottery(action)
    predicted_daly_burden_xcs = predicted_lotteries_for_actions[action]['X/C/S']['dalys']
    predicted_joint_prob_xcs = predicted_lotteries_for_actions[action]['X/C/S']['joint prob']

    predicted_daly_burden_xcccsc = predicted_lotteries_for_actions[action]['Xc/Cc/Sc']['dalys']
    predicted_joint_prob_xcccsc = predicted_lotteries_for_actions[action]['Xc/Cc/Sc']['joint prob']
    print("Action: ", action)
    check_xcs_daly_burden(lottery, predicted_daly_burden_xcs)
    check_xcs_joint_prob(lottery, predicted_joint_prob_xcs)
    check_xcccsc_daly_burden(lottery, predicted_daly_burden_xcccsc)
    check_xcccsc_joint_prob(lottery, predicted_joint_prob_xcccsc)
    return

def check_all_lotteries():
    for action in interventions:
        check_lottery(action)
    return

check_all_lotteries()

def get_ranked_state_tuples(action, idx, to_print=False):
    lottery = get_one_lottery(action)
    ranked_state_tuples = reu.get_ranked_state_tuples_one_sim(idx, lottery)
    
    if to_print:
        print("Action: ", action)
        print(ranked_state_tuples)

    return ranked_state_tuples

def get_ranked_state_tuples_one_action(action, to_print=False):
    lottery = get_one_lottery(action)
    ranked_tuples = []
    for i in range(N):   
        ranked_tuples.append(reu.get_ranked_state_tuples_one_sim(i, lottery))
    ranked_tuples = ranked_tuples
    if to_print:
        print("ranked tuples", ranked_tuples)
    return ranked_tuples

def get_probability_of_at_least_as_good(ranked_state_tuples, to_print=False):
    probability_of_at_least_as_good = []
    for i in range(N):
        probability_of_at_least_as_good.append(reu.get_probability_of_at_least_as_good_one_sim(ranked_state_tuples[i]))
    
    probability_of_at_least_as_good = probability_of_at_least_as_good
    if to_print:
        print("probability at least as good", probability_of_at_least_as_good)
    
    return probability_of_at_least_as_good

def get_reus(probability_of_at_least_as_good, to_print=False):
    reus = []
    for i in range(N):
        reus.append(reu.get_reu_one_sim(probability_of_at_least_as_good[i], 1))
    
    reus = np.array(reus)
    if to_print:
        print("reus", reus)
    return reus

def test_get_one_action_reus(action, a, to_print=False): 
    lottery = get_one_lottery(action)
    ranked_state_tuples = get_ranked_state_tuples_one_action(action)
    probability_of_at_least_as_good = get_probability_of_at_least_as_good(ranked_state_tuples)
    expected_reus = get_reus(probability_of_at_least_as_good)
    
    actual_reus = reu.get_one_action_reus(a, action, lottery)
    
    if to_print:
        print("Action: ", action)
        print("Expected REUs: ", expected_reus)
        print("Actual REUs: ", actual_reus)

        if np.array_equal(expected_reus, actual_reus):
            print("The REUs are correct.")

    return expected_reus

def test_get_all_actions_reu(a):
    reus = {}
    for action in interventions:
        reus[action] = test_get_one_action_reus(action, a)
    return reus

def test_reus_all_actions(a):
    reus, changes = reu.get_reus_all_actions(a, daly_burden_by_harm_dict, states_of_world_dict, daly_burden_by_state_dict, \
                                             joint_probs_dict, event_probs, amf_dalys_per_1000, hens_dalys_per_1000, \
                                             shrimp_slaughter_dalys_per_1000, shrimp_nh3_dalys_per_1000, \
                                             conservative_xrisk_reduction_per_bn, risky_xrisk_reduction_per_bn, to_print=False)
    expected_reus = test_get_all_actions_reu(a)
    expected_changes = {}
    expected_changes["Nothing"] = np.zeros(N)
    for action in interventions[1:]:
        print(action)
        expected_changes_action = expected_reus[action] - expected_reus["Nothing"]
        expected_changes[action] = expected_changes_action
        if np.array_equal(expected_changes_action, changes[action]):
            print("The changes in REUs are correct.")
        else:
            print("The changes in REUs are incorrect.")
            print("Expected changes: ", expected_changes_action)
            print("Actual changes: ", changes[action])

        if np.array_equal(expected_reus[action], reus[action]):
            print("The REUs are correct.")
        else:
            print("The REUs are incorrect.")
            print("Expected REUs: ", expected_reus[action])
            print("Actual REUs: ", reus[action])

    return 

test_reus_all_actions(1)