import numpy as np
class Unit:
    def __init__(self, grp_size, max_cft_pkt):
        self.mal = np.zeros((grp_size, 2))
        self.craft = np.zeros((grp_size, max_cft_pkt, 3))