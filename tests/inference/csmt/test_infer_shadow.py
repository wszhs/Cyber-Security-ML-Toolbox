import os
import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)

import numpy as np

from csmt.utils import load_nursery

(x_target, y_target), (x_shadow, y_shadow), _, _ = load_nursery(test_set=0.75)

target_train_size = len(x_target) // 2
x_target_train = x_target[:target_train_size]
y_target_train = y_target[:target_train_size]
x_target_test = x_target[target_train_size:]
y_target_test = y_target[target_train_size:]

from sklearn.ensemble import RandomForestClassifier
from csmt.estimators.classification.scikitlearn import SklearnClassifier

model = RandomForestClassifier()
model.fit(x_target_train, y_target_train)

art_classifier = SklearnClassifier(model=model,clip_values=(0,1))

print('Base model accuracy:', model.score(x_target_test, y_target_test))

from csmt.attacks.inference.membership_inference import ShadowModels
from csmt.utils import to_categorical

shadow_models = ShadowModels(art_classifier, num_shadow_models=3)

shadow_dataset = shadow_models.generate_shadow_dataset(x_shadow, to_categorical(y_shadow, 4))
(member_x, member_y, member_predictions), (nonmember_x, nonmember_y, nonmember_predictions) = shadow_dataset

# Shadow models' accuracy
# print([sm.model.score(x_target_test, y_target_test) for sm in shadow_models.get_shadow_models()])

# black_box
from csmt.attacks.inference.membership_inference import MembershipInferenceBlackBox, black_box

attack = MembershipInferenceBlackBox(art_classifier, attack_model_type="rf")
attack.fit(member_x, member_y, nonmember_x, nonmember_y, member_predictions, nonmember_predictions)