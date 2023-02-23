import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from dataclasses import dataclass
from functions import *

"""
clark flood routing method is used to calculate the outflow of the natural subcatchment
here is the eqation:
$$\frac{I_1+I_2}{\Delta{t}}-\frac{O_1+O_2}{\Delta{t}}=K(O_2 - O_1)$$

$$m_0 = \frac{0.5\Delta{t}}{K+0.5\Delta{t}}, m_1 = \frac{0.5\Delta{t}}{K+0.5\Delta{t}},m_2 = \frac{K-0.5\Delta{t}}{K+0.5\Delta{t}}$$

$$O_2 = m_0I_2+m_1I_1+m_2O_1$$

$$I_1 = I_2$$

$$O_2 = (m_0+m_1)I+m_2O_1$$
"""

@dataclass
class Storage_Constant(object):
    coeff_dict : dict

    def clark_constant(self):
    # check if the keys {"c","L","S"} are exist
        assert "c" in self.coeff_dict.keys(), "The key 'c' is not exist"
        assert "L" in self.coeff_dict.keys(), "The key 'L' is not exist"
        assert "S" in self.coeff_dict.keys(), "The key 'S' is not exist"
    # calculate the constant
        return self.coeff_dict["c"] * self.coeff_dict["L"] / np.sqrt(self.coeff_dict["S"])

    def Linsley_constant(self):
    # check if the keys {"b","L","S","A"} are exist
        assert "b" in self.coeff_dict.keys(), "The key 'b' is not exist"
        assert "L" in self.coeff_dict.keys(), "The key 'L' is not exist"
        assert "S" in self.coeff_dict.keys(), "The key 'S' is not exist"
        assert "A" in self.coeff_dict.keys(), "The key 'A' is not exist"
    # calculate the constant
        return self.coeff_dict["b"] * self.coeff_dict["L"]  * np.sqrt(self.coeff_dict["A"]) / np.sqrt(self.coeff_dict["S"])

    def Russell_constant(self):
    # check if the keys {"alpha","t_c"} are exist
        assert "alpha" in self.coeff_dict.keys(), "The key 'alpha' is not exist"
        assert "t_c" in self.coeff_dict.keys(), "The key 't_c' is not exist"
    # calculate the constant
        return self.coeff_dict["alpha"] * self.coeff_dict["t_c"]

    def Sabol_constant(self):
    # check if the keys {"t_c","L","A"} are exist
        assert "t_c" in self.coeff_dict.keys(), "The key 't_c' is not exist"
        assert "L" in self.coeff_dict.keys(), "The key 'L' is not exist"
        assert "A" in self.coeff_dict.keys(), "The key 'A' is not exist"
    # calculate the constant
        return self.coeff_dict["t_c"] / (1.46-0.0867*(self.coeff_dict["L"] ** 2) * self.coeff_dict["A"])

    def __post_init__(self):
        # Check if the required keys are present in coeff_dict
        if set(["c", "L", "S"]).issubset(self.coeff_dict.keys()):
            self.clark = self.clark_constant()
        if set(["b", "L", "S", "A"]).issubset(self.coeff_dict.keys()):
            self.Linsley = self.Linsley_constant()
        if set(["alpha", "t_c"]).issubset(self.coeff_dict.keys()):
            self.Russell = self.Russell_constant()
        if set(["t_c", "L", "A"]).issubset(self.coeff_dict.keys()):
            self.Sabol = self.Sabol_constant()

    def mean_constant(self):
        return np.mean([self.clark,self.Linsley,self.Russell,self.Sabol])


@dataclass
class Clark(object):
    time_index : np.ndarray
    al : np.ndarray
    K : float
    dt : float

    def __post_init__(self):
        self.m0 = 0.5 * self.dt / (self.K + 0.5 * self.dt)
        self.m1 = self.m0
        self.m2 = 1-self.m0

    def clark(self):
        col_3 = self.al * (self.m0 + self.m1)
        col_4 = np.zeros_like(self.time_index, dtype=float)
        col_5 = np.zeros_like(self.time_index, dtype=float)

        for i in range(1, len(self.time_index)):
            if i == 1:
                col_5[i] = col_3[i]
                continue
            col_4[i] = (1-self.m0-self.m1) * col_5[i - 1]
            col_5[i] = col_4[i] + col_3[i]
        col_6 = (col_5 + roll_and_fill(col_5, 1)) / 2
        return col_3, col_4, col_5, col_6















