import wlu_function  as wlu
import numpy as np
import squigglepy as sq
from squigglepy.numbers import K, M, B

a = [1*M]
p_a = [1]

b = [0, 1*M, 5*M]
p_b = [0.01, 0.89, 0.10]

c = [0, 1*M]
p_c = [0.89, 0.11]

d = [0, 5*M]
p_d = [0.9, 0.1]

wlu.get_table(a, weight_function=wlu.weight_function_symmetric, power=0.25, p=p_a)
wlu.get_table(b, weight_function=wlu.weight_function_symmetric, power=0.25, p=p_b)
wlu.get_table(c, weight_function=wlu.weight_function_symmetric, power=0.25, p=p_c)
wlu.get_table(d, weight_function=wlu.weight_function_symmetric, power=0.25, p=p_d)
