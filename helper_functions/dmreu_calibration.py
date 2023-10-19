### Ask the user questions to calibrate their level of risk aversion

def user_dmreu_level():
    '''
    Asks the user at what probability they're indifferent between a guaranteed averting 10,000 DALYs 
        and a X% chance of averting 1,000,000 DALYs. This probability is returned to the user and used in 
        subsequent risk-weighted expected utility calculations.
    '''
    guarantee = 10**4
    risky = 10**6
    prob_win = float(input("You're given the choice between two lotteries. If you choose lottery A, you save {} DALYs guaranteed. If you choose lottery B, you have an X% chance of averting {} DALYs, otherwise you get nothing. What is the smallest value of X such that you choose lottery B over lottery A?".format(guarantee, risky)))

    print("In a choice between averting 10,000 DALYs with certainty or taking a bet with an X% chance of averting 1M DALYs, you said that X has to be at least {} for you to choose the gamble.".format(prob_win*100))

    return prob_win
