import sympy as sy



x, t, z, nu, K = sy.symbols("x t z nu K")
'''
O1 = Function("O1")
eq1 = Eq(O1(t),O1(t).diff(t,1)*K)
deq1 = dsolve(eq1,O1(t),ics={O1(0):(1/K)})
O1 = deq1.rhs

O2 = Function("O2")
eq2 = Eq(O2(t)-O1,O2(t).diff(t,1)*K)
deq2 = dsolve(eq2, O2(t), ics = {O2(0):0})
O2 = deq2.rhs

O3 = Function("O3")
eq3 = Eq(O3(t)-O2,O3(t).diff(t,1)*K)
deq3 = dsolve(eq3, O3(t), ics = {O3(0):0})
O3 = deq3.rhs

i want to repeat this calculate process while O10 how can i make this code more simple?
'''

O1 = sy.Function("O1")
eq1 = sy.Eq(-O1(t), O1(t).diff(t, 1) * K)
deq1 = sy.dsolve(eq1, O1(t), ics={O1(0): (1 / K)})
O1 = deq1.rhs

O2 = sy.Function("O2")
eq2 = sy.Eq(O2(t) - O1, O2(t).diff(t, 1) * K)
deq2 = sy.dsolve(eq2, O2(t), ics={O2(0): 0})
O2 = deq2.rhs

O3 = sy.Function("O3")
eq3 = sy.Eq(O3(t) - O2, O3(t).diff(t, 1) * K)
deq3 = sy.dsolve(eq3, O3(t), ics={O3(0): 0})
O3 = deq3.rhs

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
