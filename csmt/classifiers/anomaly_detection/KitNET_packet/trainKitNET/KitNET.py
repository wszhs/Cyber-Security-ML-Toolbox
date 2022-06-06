import numpy as np
import csmt.classifiers.anomaly_detection.KitNET_packet.trainKitNET.dA as AE
import csmt.classifiers.anomaly_detection.KitNET_packet.trainKitNET.corClust as CC
import pickle

# This class represents a KitNET machine learner.
# KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders.
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
# For licensing information, see the end of this document

class KitNET:
    #n: the number of features in your input dataset (i.e., x \in R^n)
    #m: the maximum size of any autoencoder in the ensemble layer
    #AD_grace_period: the number of instances the network will learn from before producing anomaly scores
    #FM_grace_period: the number of instances which will be taken to learn the feature mapping. If 'None', then FM_grace_period=AM_grace_period
    #learning_rate: the default stochastic gradient descent learning rate for all autoencoders in the KitNET instance.
    #hidden_ratio: the default ratio of hidden to visible neurons. E.g., 0.75 will cause roughly a 25% compression in the hidden layer.
    #feature_map: One may optionally provide a feature map instead of learning one. The map must be a list,
    #           where the i-th entry contains a list of the feature indices to be assingned to the i-th autoencoder in the ensemble.
    #           For example, [[2,5,3],[4,0,1],[6,7]]
    def __init__(self,model,n,max_autoencoder_size=10,FM_grace_period=None,AD_grace_period=10000,learning_rate=0.1,hidden_ratio=0.75, feature_map = None):
        self.model=model
        # print(model_save_path)
        # self.f = '...'

        # Parameters:
        self.AD_grace_period = AD_grace_period
        if FM_grace_period is None:
            self.FM_grace_period = AD_grace_period
        else:
            self.FM_grace_period = FM_grace_period
        if max_autoencoder_size <= 0:
            self.m = 1
        else:
            self.m = max_autoencoder_size
        self.lr = learning_rate
        self.hr = hidden_ratio
        self.n = n

        # Variables
        self.n_trained = 0 # the number of training instances so far
        self.n_executed = 0 # the number of executed instances so far
        self.v = feature_map
        if self.v is None:
            print("Feature-Mapper: train-mode, Anomaly-Detector: off-mode")
        else:
            self.__createAD__()
            print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
        self.FM = CC.corClust(self.n) #incremental feature cluatering for the feature mapping process
        self.ensembleLayer = []
        self.outputLayer = None

    #If FM_grace_period+AM_grace_period has passed, then this function executes KitNET on x. Otherwise, this function learns from x.
    #x: a numpy array of length n
    #Note: KitNET automatically performs 0-1 normalization on all attributes.
    def process(self,x):
        if self.n_trained >= self.FM_grace_period + self.AD_grace_period: #If both the FM and AD are in execute-mode
            # print(self.n_trained,self.FM_grace_period,self.AD_grace_period)
            # return self.execute(x)
            # print(111)
            return 0.0
        else:
            # print(222)
            return self.train(x)
            # return 0.0

    #force train KitNET on x
    #returns the anomaly score of x during training (do not use for alerting)
    def train(self,x):
        if self.n_trained <= self.FM_grace_period and self.v is None: #If the FM is in train-mode, and the user has not supplied a feature mapping
            #update the incremetnal correlation matrix
            self.FM.update(x)
            if self.n_trained == self.FM_grace_period: #If the feature mapping should be instantiated
                self.v = self.FM.cluster(self.m)
                # for i in self.v:
                #     print(i)
                # exit(-1)
                # pickle.dump(self.v,self.f)
                self.model.append(self.v)
                # print("## mapping(v) has been saved in file...")
                # f.close()
                # exit(-1)
                self.__createAD__()
                # print("The Feature-Mapper found a mapping: "+str(self.n)+" features to "+str(len(self.v))+" autoencoders.")
                # print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
            train_rmse = 0.
        else: #train
            ## Ensemble Layer
            S_l1 = np.zeros(len(self.ensembleLayer))
            for a in range(len(self.ensembleLayer)):
                # make sub instance for autoencoder 'a'
                xi = x[self.v[a]]
                # print(xi,type(xi))
                # exit(-1)
                S_l1[a] = self.ensembleLayer[a].train(xi)
            ## OutputLayer
            train_rmse = self.outputLayer.train(S_l1)
            # print(train_rmse)
            # return train_rmse
            # print(self.n_trained)
            # print(self.AD_grace_period+self.FM_grace_period-1)

            if self.n_trained == self.AD_grace_period+self.FM_grace_period-1:
                # pickle.dump(self.ensembleLayer, self.f)
                self.model.append(self.ensembleLayer)
                # print("## ensembleLayer has been saved in file...")
                # pickle.dump(self.outputLayer, self.f)
                self.model.append(self.outputLayer)
                # print("## outputLayer has been saved in file...")
                # print("Feature-Mapper: execute-mode, Anomaly-Detector: execute-mode")
                # exit(-1)

        self.n_trained += 1
        return train_rmse

    #force execute KitNET on x
    def execute(self,x):
        if self.v is None:
            raise RuntimeError('KitNET Cannot execute x, because a feature mapping has not yet been learned or provided. Try running process(x) instead.')
        else:
            self.n_executed += 1
            ## Ensemble Layer
            S_l1 = np.zeros(len(self.ensembleLayer))
            for a in range(len(self.ensembleLayer)):
                # make sub inst
                xi = x[self.v[a]]
                # print(xi,type(xi))
                # exit(-1)
                S_l1[a] = self.ensembleLayer[a].execute(xi)
            ## OutputLayer
            return self.outputLayer.execute(S_l1)

    def __createAD__(self):
        # construct ensemble layer
        for map in self.v:
            params = AE.dA_params(n_visible=len(map), n_hidden=0, lr=self.lr, corruption_level=0, gracePeriod=0, hiddenRatio=self.hr)
            self.ensembleLayer.append(AE.dA(params))

        # construct output layer
        params = AE.dA_params(len(self.v), n_hidden=0, lr=self.lr, corruption_level=0, gracePeriod=0, hiddenRatio=self.hr)
        self.outputLayer = AE.dA(params)

# Copyright (c) 2017 Yisroel Mirsky
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.























