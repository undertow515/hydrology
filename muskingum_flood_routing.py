import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Calculation procedure by muskingum flood routing method

'''
$$C_0 = \frac{-(Kx-0.5\Delta{t})}{K-Kx+0.5\Delta{t}}, C_1 = \frac{(Kx+0.5\Delta{t})}{K-Kx+0.5\Delta{t}}, C_2 = \frac{(K-Kx-0.5\Delta{t})}{K-Kx+0.5\Delta{t}}$$
$$C_0 + C_1 + C_2 = 1$$
$$C_0, C_1, C_2 > 0$$
'''

timedelta = 6  # hr

# Input data
# time(hr)
t = np.arange(0, timedelta * 6 + 1, timedelta)
# Inflow(m3/s)
Inflow = np.array([10, 30, 68, 50, 40, 31, 23])
# define initial Coefficient

def initial_muskingum_coefficient(timedelta: float, K: float, x: float):
    return np.array([-(K * x - 0.5 * timedelta) / (K - K * x + 0.5 * timedelta),
                     (K * x + 0.5 * timedelta) / (K - K * x + 0.5 * timedelta),
                     (K - K * x - 0.5 * timedelta) / (K - K * x + 0.5 * timedelta)])

def muskingum_flood_routing(Inflow: np.ndarray, coeff_array: np.ndarray):
    # Outflow(m3/s)
    Outflow = np.zeros_like(Inflow, dtype=float)
    Outflow[0] = Inflow[0]
    for i in range(1, len(Inflow)):
        Outflow[i] = coeff_array[0] * Inflow[i] + coeff_array[1] * Inflow[i - 1] + coeff_array[2] * Outflow[i - 1]

    return Outflow


x = 0.13
K = 11
coeff_array = initial_muskingum_coefficient(timedelta, K, x)

Outflow = muskingum_flood_routing(Inflow, coeff_array)

# make dataframe by pandas
df = pd.DataFrame({'Inflow': Inflow, 'Outflow': Outflow}, index=t)

# 2 x 2 subplot xI+(1-x)O by (Inflow - Outflow).cumsum() x is 0.03 to 0.23 step = 0.05
x = np.arange(0.1, 1, 0.2)
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        coeff_array = initial_muskingum_coefficient(timedelta, K, x[i * 2 + j])
        Outflow = muskingum_flood_routing(Inflow, coeff_array)
        df2 = pd.DataFrame({'Inflow': Inflow, 'Outflow': Outflow}, index=t)
        ax[i, j].plot((df2['Inflow'] - df2['Outflow']).cumsum(), x[i * 2 + j] * df2["Inflow"] + (1 - x[i * 2 + j]) * df2["Outflow"])
        # set ylimit
        ax[i, j].set_ylim(0, 50)
        # add text to subplot using np.corrcoef
        ax[i, j].text(0.5, 0.9, 'r = {:.2f}'.format(np.corrcoef((df2['Inflow'] - df2['Outflow']).cumsum(), x[i * 2 + j] * df2["Inflow"] + (1 - x[i * 2 + j]) * df2["Outflow"])[0, 1]))
        ax[i, j].set_title('x = {}'.format(x[i * 2 + j]))
        ax[i, j].set_xlabel('S')
        ax[i, j].set_ylabel('xI+(1-x)O')
plt.tight_layout()
plt.close()
# plt.show()

from sklearn.linear_model import LinearRegression

x = 0.13
line_fitter = LinearRegression()
line_fitter.fit((np.array(df['Inflow'] - df['Outflow']).cumsum()).reshape(-1, 1), np.array(x * df["Inflow"] + (1 - x) * df["Outflow"]))

# plot linear regression
plt.plot((np.array(df['Inflow'] - df['Outflow']).cumsum()), line_fitter.predict((np.array(df['Inflow'] - df['Outflow']).cumsum()).reshape(-1, 1)))
plt.plot((np.array(df['Inflow'] - df['Outflow']).cumsum()), x * df["Inflow"] + (1 - x) * df["Outflow"])

# plot slope and intercept
plt.text(10, 10, 'slope = {:.2f}'.format(line_fitter.coef_[0]))
plt.text(0, 20, 'intercept = {:.2f}'.format(line_fitter.intercept_))

plt.xlabel('S')
plt.ylabel('xI+(1-x)O')
plt.title('Linear Regression')
plt.tight_layout()
plt.show()






