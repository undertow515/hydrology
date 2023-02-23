import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class Muskingum:

    def __init__(self, timedelta, K, x):
        self.timedelta = timedelta
        self.K = K
        self.x = x

        self.coeff_array = self.initial_muskingum_coefficient()

    def initial_muskingum_coefficient(self):
        return np.array([
            -(self.K * self.x - 0.5 * self.timedelta) / (self.K - self.K * self.x + 0.5 * self.timedelta),
            (self.K * self.x + 0.5 * self.timedelta) / (self.K - self.K * self.x + 0.5 * self.timedelta),
            (self.K - self.K * self.x - 0.5 * self.timedelta) / (self.K - self.K * self.x + 0.5 * self.timedelta)
        ])

    def muskingum_flood_routing(self, Inflow):
        Outflow = np.zeros_like(Inflow, dtype=float)
        Outflow[0] = Inflow[0]
        for i in range(1, len(Inflow)):
            Outflow[i] = self.coeff_array[0] * Inflow[i] + self.coeff_array[1] * Inflow[i - 1] + self.coeff_array[2] * \
                         Outflow[i - 1]
        return Outflow

    def plot_xI_plus_1_minus_xO(self, Inflow, Outflow, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot((Inflow - Outflow).cumsum(), self.x * Inflow + (1 - self.x) * Outflow)
        ax.set_xlabel('S')
        ax.set_ylabel('xI + (1 - x)O')
        ax.set_ylim(0, 50)
        ax.set_title('x = {}'.format(self.x))
        ax.text(0.5, 0.9, 'r = {:.2f}'.format(
            np.corrcoef((Inflow - Outflow).cumsum(), self.x * Inflow + (1 - self.x) * Outflow)[0, 1]))

    def plot_linear_regression(self, Inflow, Outflow):
        line_fitter = LinearRegression()
        line_fitter.fit((Inflow - Outflow).cumsum().reshape(-1, 1), self.x * Inflow + (1 - self.x) * Outflow)
        plt.plot((Inflow - Outflow).cumsum(), line_fitter.predict((Inflow - Outflow).cumsum().reshape(-1, 1)))
        plt.plot((Inflow - Outflow).cumsum(), self.x * Inflow + (1 - self.x) * Outflow)
        plt.xlabel('S')
        plt.ylabel('xI + (1 - x)O')
        plt.title('Linear Regression')
        plt.tight_layout()
        plt.text(10, 10, 'slope = {:.2f}'.format(line_fitter.coef_[0]))
        plt.text(0, 20, 'intercept = {:.2f}'.format(line_fitter.intercept_))

