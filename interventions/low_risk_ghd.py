### DALYs/$1000 for low-risk GHD interventions (Against Malaria Foundation)

import numpy as np
import pandas as pd
import squigglepy as sq
from squigglepy.numbers import K, M, B

N = 6*M

def amf_dalys_per_1000(to_print=False):
    '''
    The DALYs averted per $1000 spent on Against Malaria Foundation. We add a 3% probability that AMF is doing harm to the world,
        but that the cost to increase the DALY burden of the world is 3x the cost to decrease it. 
    '''
    cost_per_daly_good = sq.norm(35, 70, lclip=30, rclip=85)
    cost_per_daly_bad = -3*cost_per_daly_good
    cost_per_daly = sq.sample(sq.mixture([cost_per_daly_good, cost_per_daly_bad], [0.97, 0.03]), N)

    dalys_per_1000_amf = 1000/cost_per_daly

    if to_print:
        print("Mean DALYs per $1000: {}".format(np.mean(dalys_per_1000_amf)))
        print("Percentiles:")
        print(sq.get_percentiles(dalys_per_1000_amf))
    return dalys_per_1000_amf


