from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import numpy as np


def multiclass_node_classification_eval(x, y, ratio=0.2, rnd=2018):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ratio, random_state=rnd)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')

    return macro_f1, micro_f1


def node_multiclass_classification(x, y):
    for i in np.arange(0.1, 1, 0.1):
        macro_f1_avg = 0.0
        micro_f1_avg = 0.0
        for j in range(10):
            rnd = np.random.randint(2018)
            macro_f1, micro_f1 = multiclass_node_classification_eval(x, y, i, rnd)
            macro_f1_avg += macro_f1
            micro_f1_avg += micro_f1
        macro_f1_avg /= 10
        micro_f1_avg /= 10
        print('%.2f%% test data, macro_f1 = %f, micro_f1 = %f' % (i * 100, macro_f1_avg, micro_f1_avg))
