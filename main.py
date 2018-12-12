import numpy as np
import sklearn
# import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

FILENAME = "wifi_localization.txt"

lines = [line.rstrip('\n').split('\t') for line in open(FILENAME)]
lines = [[int(x) for x in line] for line in lines]

dataset = {}
dataset['data'] = np.array([line[:-1] for line in lines])
dataset['target'] = np.array([line[-1] for line in lines])

print(dataset['target'].shape)