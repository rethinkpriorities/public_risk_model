import wlu_function  as wlu
import numpy as np
import squigglepy as sq
from squigglepy.numbers import K, M, B

a = [90]
p_a = [1]

b = [0, 1*K, -1*K]
p_b = [0.89, 0.10, 0.01]

c = [90*M]
p_c = [1]

d = [0, 1*B, -1*B]
p_d = [0.89, 0.1, 0.01]

print("c = ", 0)
print(wlu.get_weighted_linear_utility(a,  power = 0, weight_function=wlu.weight_function_symmetric, p = p_a, to_print=True))
print(wlu.get_weighted_linear_utility(b, power =  0,  weight_function=wlu.weight_function_symmetric,  p= p_b, to_print=True))
print(wlu.get_weighted_linear_utility(c,  power = 0, weight_function=wlu.weight_function_symmetric,  p= p_c, to_print=True))
print(wlu.get_weighted_linear_utility(d,  power = 0, weight_function=wlu.weight_function_symmetric, p= p_d, to_print=True))
print("\n")

print("c = ", 0.01)
print(wlu.get_weighted_linear_utility(a,  power = 0.01, weight_function=wlu.weight_function_symmetric, p = p_a, to_print=True))
print(wlu.get_weighted_linear_utility(b, power =  0.01,  weight_function=wlu.weight_function_symmetric,  p= p_b, to_print=True))
print(wlu.get_weighted_linear_utility(c,  power = 0.01, weight_function=wlu.weight_function_symmetric, p= p_c, to_print=True))
print(wlu.get_weighted_linear_utility(d,  power = 0.01, weight_function=wlu.weight_function_symmetric, p= p_d, to_print=True))
print("\n")

print("c = ", 0.05)
print(wlu.get_weighted_linear_utility(a,  power = 0.05, weight_function=wlu.weight_function_symmetric, p = p_a, to_print=True))
print(wlu.get_weighted_linear_utility(b, power =  0.05,  weight_function=wlu.weight_function_symmetric,  p= p_b, to_print=True))
print(wlu.get_weighted_linear_utility(c,  power = 0.05, weight_function=wlu.weight_function_symmetric, p= p_c, to_print=True))
print(wlu.get_weighted_linear_utility(d,  power = 0.05, weight_function=wlu.weight_function_symmetric, p= p_d, to_print=True))
print("\n")

print("c = ", 0.10)
print(wlu.get_weighted_linear_utility(a,  power = 0.1, weight_function=wlu.weight_function_symmetric, p = p_a, to_print=True))
print(wlu.get_weighted_linear_utility(b, power =  0.1,  weight_function=wlu.weight_function_symmetric,  p= p_b, to_print=True))
print(wlu.get_weighted_linear_utility(c,  power = 0.1, weight_function=wlu.weight_function_symmetric, p= p_c, to_print=True))
print(wlu.get_weighted_linear_utility(d,  power = 0.1, weight_function=wlu.weight_function_symmetric, p= p_d, to_print=True))
print("\n")

print("c = ", 0.15)
print(wlu.get_weighted_linear_utility(a,  power = 0.15, weight_function=wlu.weight_function_symmetric, p = p_a, to_print=True))
print(wlu.get_weighted_linear_utility(b, power =  0.15, weight_function=wlu.weight_function_symmetric,  p= p_b, to_print=True))
print(wlu.get_weighted_linear_utility(c,  power = 0.15, weight_function=wlu.weight_function_symmetric, p= p_c, to_print=True))
print(wlu.get_weighted_linear_utility(d,  power = 0.15, weight_function=wlu.weight_function_symmetric, p= p_d, to_print=True))
print("\n")

print("c = ", 0.20)
print(wlu.get_weighted_linear_utility(a,  power = 0.20, weight_function=wlu.weight_function_symmetric, p = p_a, to_print=True))
print(wlu.get_weighted_linear_utility(b, power =  0.20, weight_function=wlu.weight_function_symmetric,  p= p_b, to_print=True))
print(wlu.get_weighted_linear_utility(c,  power = 0.20, weight_function=wlu.weight_function_symmetric, p= p_c, to_print=True))
print(wlu.get_weighted_linear_utility(d,  power = 0.20, weight_function=wlu.weight_function_symmetric, p= p_d, to_print=True))
print("\n")

print("c = ", 0.25)
print(wlu.get_weighted_linear_utility(a,  power = 0.25, weight_function=wlu.weight_function_symmetric, p = p_a, to_print=True))
print(wlu.get_weighted_linear_utility(b, power =  0.25, weight_function=wlu.weight_function_symmetric, p= p_b, to_print=True))
print(wlu.get_weighted_linear_utility(c,  power = 0.25, weight_function=wlu.weight_function_symmetric, p= p_c, to_print=True))
print(wlu.get_weighted_linear_utility(d,  power = 0.25, weight_function=wlu.weight_function_symmetric, p= p_d, to_print=True))
print("\n")