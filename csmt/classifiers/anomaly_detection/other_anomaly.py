from csmt.classifiers.abstract_model import AbstractModel
from csmt.estimators.classification.anomaly_classifier import AnomalyClassifeir
class AbOCSVM(AbstractModel):

    def __init__(self,input_size,output_size):
        from csmt.classifiers.anomaly_detection.pyod.models.ocsvm import OCSVM
        model=OCSVM()
        self.classifier=AnomalyClassifeir(model=model,nb_features=input_size, nb_classes=output_size,clip_values=(0,1),contamination=0.1)

class AbHBOS(AbstractModel):

    def __init__(self,input_size,output_size):
        from csmt.classifiers.anomaly_detection.pyod.models.hbos import HBOS
        model=HBOS()
        self.classifier=AnomalyClassifeir(model=model,nb_features=input_size, nb_classes=output_size,clip_values=(0,1),contamination=0.1)

class AbVAE(AbstractModel):

    def __init__(self,input_size,output_size):
        from csmt.classifiers.anomaly_detection.pyod.models.vae import VAE
        model= VAE(epochs=30, contamination=0.1,verbose=0)
        self.classifier=AnomalyClassifeir(model=model,nb_features=input_size, nb_classes=output_size,clip_values=(0,1),contamination=0.1)