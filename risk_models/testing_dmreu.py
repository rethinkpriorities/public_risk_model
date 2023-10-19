### Testing dmreu_function.py. Not used anywhere else. 

import dmreu_function as dmreu
import squigglepy as sq

x = [-100, 0, 100]
px = [0.1, 0.8, 0.1]

## expect to get -18 = -100*1**2 + (0--100)*0.8**2 + (100-0)*0.1**2
dmreu.get_risk_weighted_utility(x, 2, p = px, to_print = True)