import numpy as np

class Settings():
    def __init__(self):

        #-------- map realization ----------
        self.nside = 1024
        # self.nside = 2048
        self.samples = 4000

        #---------- scan stragegy ---------
        self.NET = 350  # uk.sqrt(s)
        self.theta = 80 # deg, zenith angle= theta, and elevation = 90-theta.

        #---------- Ali location ---------