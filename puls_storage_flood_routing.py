import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class Puls:
    def __init__(self, X, y, inflow):
        self.X = X
        self.y = y
        self.inflow = inflow
        self.line_fitter = LinearRegression()
        self.line_fitter.fit(self.X.reshape(-1, 1), self.y)
        self.outflow = [self.inflow[0]]
        self.s1_plus = []
        self.s1_minus = []
        self.s2_plus = []

    def linear_storage(self, x):
        return float(self.line_fitter.coef_) * x + float(self.line_fitter.intercept_)

    def inverse_linear_storage(self, y):
        return (y - float(self.line_fitter.intercept_)) / float(self.line_fitter.coef_)

    def run(self):
        for i in range(len(self.inflow) - 1):
            self.s1_plus.append(self.linear_storage(self.outflow[-1]))
            self.s1_minus.append(self.s1_plus[-1] - self.outflow[-1] * 2)
            self.s2_plus.append(self.s1_minus[-1] + self.inflow[i] + self.inflow[i + 1])
            self.outflow.append(self.inverse_linear_storage(self.s2_plus[-1]))

    def get_dataframe(self):
        hours_passed = pd.date_range(start='2020-01-01', periods=len(self.inflow), freq='6H')
        df = pd.DataFrame({'Inflow': self.inflow, 'Outflow': self.outflow}, index=hours_passed)
        return df

# test

X = np.array([0, 17, 48.08, 88.33, 136, 190.07, 249.85])
y = np.array([0, 572.56, 1159.18, 1754.99, 2358.22, 2967.85, 3583.17])
inflow = np.hstack(([17,20,50,100,130,150,140,110,90,70,50,30,20],np.ones(12)*17))

puls = Puls(X, y, inflow)
puls.run()
df = puls.get_dataframe()

print(df)





















