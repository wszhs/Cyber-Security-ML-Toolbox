
from csmt.classifiers.abstract_model import AbstractModel

class DeepForest(AbstractModel):
    def __init__(self,input_size,output_size):
        from deepforest import CascadeForestClassifier
        from csmt.estimators.classification.ensemble_tree import EnsembleTree
        model=CascadeForestClassifier(random_state=1,verbose=0)
        self.classifier=EnsembleTree(model=model,nb_features=input_size, nb_classes=output_size,clip_values=(0,1))
