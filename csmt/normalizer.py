import numpy as np
class Normalizer:
    def __init__(self, 
            dim, 
            normer="minmax",
            online_minmax=False): # whether fit_transform online (see Kitsune), *available only for normer="minmax"

        self.dim = dim # feature dimensionality
        self.normer = normer
        if self.normer == 'minmax':
            self.online_minmax = online_minmax
            self.norm_max = [-np.Inf] * self.dim
            self.norm_min = [np.Inf] * self.dim
        else:
            raise NotImplementedError # Implement other Normalizer here
        
    def fit_transform(self,train_feat):
        if self.normer == 'minmax':
            return self._minmax_fit_transform(train_feat)
        else:
            raise NotImplementedError # Implement other Normalizer here

    def transform(self,feat):
        if self.normer == 'minmax':
            return self._minmax_transform(feat)
        else:
            raise NotImplementedError # Implement other Normalizer here

    def restore(self,feat):
        if self.normer == 'minmax':
            return self._minmax_restore(feat)
        else:
            raise NotImplementedError # Implement other Normalizer here
        
    def _minmax_fit_transform(self,train_feat):
        if not self.online_minmax:
            self.norm_min = np.min(train_feat,axis=0)
            self.norm_max = np.max(train_feat,axis=0)
            norm_feat = (train_feat - self.norm_min) / (self.norm_max-self.norm_min+1e-10)
            return norm_feat
        else:
            norm_feat = []
            self.norm_max, self.norm_min = np.asarray(self.norm_max), np.asarray(self.norm_min)
            for i in range(len(train_feat)):
                x = train_feat[i]
                self.norm_max[x>self.norm_max] = x[x>self.norm_max]
                self.norm_min[x<self.norm_min] = x[x<self.norm_min]
                norm_feat.append((x - self.norm_min) / (self.norm_max-self.norm_min+1e-10))
            return np.asarray(norm_feat)

    def _minmax_transform(self, feat):
        norm_feat = (feat - self.norm_min) / (self.norm_max-self.norm_min+1e-10)
        return norm_feat

    def _minmax_restore(self, feat):
        denorm_feat = feat * (self.norm_max-self.norm_min+1e-10) + self.norm_min
        return denorm_feat
    

