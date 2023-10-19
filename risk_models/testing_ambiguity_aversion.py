### This is just a test of the ambiguity_aversion.py functions

import numpy as np
import ambiguity_aversion as aa
import squigglepy as sq
from squigglepy.numbers import K, M, B  

causes = ["A", "B", "C", "D"]

eus_dict = {'A': sq.sample(sq.norm(-10, 10), 1*K),
            'B': sq.sample(sq.norm(0, 10), 1*K),
            'C': sq.sample(sq.norm(-10, 20), 1*K),
            'D': sq.sample(sq.norm(-20, 10), 1*K)}

aaev_df = aa.make_aaev_dataframe(eus_dict, causes, to_print=False)
print(aaev_df)