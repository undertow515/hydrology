import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

@dataclass
class RRL(object):
    """
    RRL is kind of rainfall-runoff flood routing model
    """
    time_index:np.array
    rainfall_intensity:np.array
    al:List[float]


    @staticmethod
    def area_ri(shape:np.shape,n:int,ri:np.array,al:List[float])->np.array:
        # ri means rainfall intensity
        # shape is the shape of time index
        # n is the number of area
        # al is the list of area(m^2)
        r = np.zeros(shape)
        r[n-1:n-1+len(ri)] = ri*al[n-1]
        return r

    def __post_init__(self):
        self.A_R = np.array([RRL.area_ri(self.time_index.shape,i,self.rainfall_intensity,self.al) for i in range(1,len(self.al)+1)])
        self.A_R_D= {"A{}".format(i+1):j for i,j in enumerate(self.A_R)}
        self.df = pd.DataFrame(self.A_R_D, index= self.time_index)

    def plotting(self):
        self.df["sumRA"] = self.df.sum(axis=1)
        self.df["I_i"] = self.df["sumRA"]*0.2778
        self.df["I_i"].plot()
        plt.show()

# use example

# time_index = np.array([i*5 for i in range(11)])
# rainfall_intensity = np.array([0,33,22,16,10,7])
# al = [0.4,0.85,1.25,0.7,0.2]
# rrl = RRL(time_index,rainfall_intensity,al)
# rrl.plotting()








