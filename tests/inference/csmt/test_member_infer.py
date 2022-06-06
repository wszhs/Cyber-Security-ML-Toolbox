
import sys
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
sys.path.append(ROOT_PATH)

from csmt.utils import load_nursery

(x_train, y_train), (x_test, y_test), _, _ = load_nursery(test_set=0.5)
print(x_train.shape)

from sklearn.ensemble import RandomForestClassifier
from csmt.estimators.classification.scikitlearn import SklearnClassifier

model = RandomForestClassifier()
model.fit(x_train, y_train)

art_classifier = SklearnClassifier(model=model,clip_values=(0,1))

print('Base model accuracy: ', model.score(x_test, y_test))

#Rule-based attack
import numpy as np
from csmt.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased

attack = MembershipInferenceBlackBoxRuleBased(art_classifier)

# infer attacked feature
inferred_train = attack.infer(x_train, y_train)
inferred_test = attack.infer(x_test, y_test)

# check accuracy
train_acc = np.sum(inferred_train) / len(inferred_train)
test_acc = 1 - (np.sum(inferred_test) / len(inferred_test))
acc = (train_acc * len(inferred_train) + test_acc * len(inferred_test)) / (len(inferred_train) + len(inferred_test))
# print(train_acc)
# print(test_acc)
# print(acc)

def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1
    
    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall

# rule-based
print(calc_precision_recall(np.concatenate((inferred_train, inferred_test)), 
                            np.concatenate((np.ones(len(inferred_train)), np.zeros(len(inferred_test))))))

# black-box attack
from csmt.attacks.inference.membership_inference import MembershipInferenceBlackBox, black_box

attack_train_ratio = 0.5
attack_train_size = int(len(x_train) * attack_train_ratio)
attack_test_size = int(len(x_test) * attack_train_ratio)

bb_attack = MembershipInferenceBlackBox(art_classifier)

# train attack model
bb_attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
              x_test[:attack_test_size], y_test[:attack_test_size])

# get inferred values
inferred_train_bb = bb_attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
inferred_test_bb = bb_attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])
# check accuracy
train_acc = np.sum(inferred_train_bb) / len(inferred_train_bb)
test_acc = 1 - (np.sum(inferred_test_bb) / len(inferred_test_bb))
acc = (train_acc * len(inferred_train_bb) + test_acc * len(inferred_test_bb)) / (len(inferred_train_bb) + len(inferred_test_bb))
print(train_acc)
print(test_acc)
print(acc)
# black-box
print(calc_precision_recall(np.concatenate((inferred_train_bb, inferred_test_bb)), 
                            np.concatenate((np.ones(len(inferred_train_bb)), np.zeros(len(inferred_test_bb))))))