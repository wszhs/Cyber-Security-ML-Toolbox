import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics

# Importing LimeTextExplainer (aix360 sytle)
from csmt.Interpretability.aix360.algorithms.lime import LimeTextExplainer

from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
# making class names shorter
class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:]) for x in newsgroups_train.target_names]
class_names[3] = 'pc.hardware'
class_names[4] = 'mac.hardware'

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=.01)
nb.fit(train_vectors, newsgroups_train.target)

pred = nb.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='weighted')

from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, nb)

print(c.predict_proba([newsgroups_test.data[0]]).round(3))

limeexplainer = LimeTextExplainer(class_names=class_names)
print(type(limeexplainer))

idx = 1340
# aix360 style for explaining input instances
exp = limeexplainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6, labels=[0, 17])
print('Document id: %d' % idx)
print('Predicted class =', class_names[nb.predict(test_vectors[idx]).reshape(1,-1)[0,0]])
print('True class: %s' % class_names[newsgroups_test.target[idx]])

print ('Explanation for class %s' % class_names[0])
print ('\n'.join(map(str, exp.as_list(label=0))))
print ()
print ('Explanation for class %s' % class_names[17])
print ('\n'.join(map(str, exp.as_list(label=17))))