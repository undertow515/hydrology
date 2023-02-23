import numpy as np
import sympy as sy
from typing import List
def roll_and_fill(arr, shift:int) -> np.ndarray:
    x = np.roll(arr, shift=shift)
    x[:shift] = arr[0]
    return x

x, t, z, nu, K = sy.symbols("x t z nu K")
# define function which is used to calculate the O_n
# O_n - O_n-1 = K * dO_n/dt
def calculate_O_n(O_n_1):
    # define the function
    O_n = sy.Function("O_n")
    # define the equation
    eq = sy.Eq(O_n(t) - O_n_1, O_n(t).diff(t, 1) * K)
    # solve the equation
    deq = sy.dsolve(eq, O_n(t), ics={O_n(0): 0})
    # return the result
    return deq.rhs

'''
O4 = calculate_O_n(O3)
O5 = calculate_O_n(O4)
O6 = calculate_O_n(O7)
O7 = calculate_O_n(O6) and so on
'''

# using calculate_O_n function, make recursive function

def calculate_O_n_recursive(O_n_1, n):
    # if n is 1, then return O_n_1
    if n == 1: return O_n_1
    # else, return the result of the recursive function
    else: return calculate_O_n_recursive(calculate_O_n(O_n_1), n - 1)

# now we can calculate the O_n

def total_output(arr, l:np.ndarray) -> np.ndarray:
    # roll_and_fill(arr,1)*l[0]
    # roll_and_fill(arr,2)*l[1]
    # roll_and_fill(arr,3)*l[2]
    return np.sum([roll_and_fill(arr,i)*l[i] for i in range(len(l))], axis=0)

def make_simple_al(arr, time_index) -> np.ndarray:
    x = np.zeros_like(time_index, dtype=float)
    x[0:len(arr)] = arr
    return x

