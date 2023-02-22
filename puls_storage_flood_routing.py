import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([0, 17, 48.08, 88.33, 136, 190.07, 249.85])
y = np.array([0, 572.56, 1159.18, 1754.99, 2358.22, 2967.85, 3583.17])
S1_plus = []
S1_minus = []
S2_plus = []
Inflow = np.hstack(([17,20,50,100,130,150,140,110,90,70,50,30,20],np.ones(12)*17))


line_fitter = LinearRegression()
line_fitter.fit(X.reshape(-1, 1), y)

def linear_storage(x : float):
    return float(line_fitter.coef_)*x + float(line_fitter.intercept_)
print(linear_storage(40))
def inverse_linear_storage(y : float):
    return (y - float(line_fitter.intercept_))/float(line_fitter.coef_)

Outflow = [17]

for i in range(len(Inflow)-1):
    S1_plus.append(linear_storage(Outflow[-1]))
    S1_minus.append(S1_plus[-1]-Outflow[-1]*2)
    S2_plus.append(S1_minus[-1]+Inflow[i]+Inflow[i+1])
    Outflow.append(inverse_linear_storage(S2_plus[-1]))

hours_passed = pd.date_range(start='2020-01-01', periods=len(Inflow), freq='6H')
df = pd.DataFrame({'Inflow': Inflow, 'Outflow': Outflow}, index=hours_passed)
























