# examples are training examples
# target_attribute is the attribute whose value is to be predicted by the tree
# attributes is a list of other attributes that may be tested by the learned decision tree

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from tree_module import *
from myid3 import *
from myc45 import *
import math
import copy

pd.options.mode.chained_assignment = None
df = pd.read_csv('iris.csv', sep=',')
attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target = 'species'

c45 = myC45(df, target, attributes)
# c45.id3.tree_.export_tree()
# c45.id3.tree_.save_to_file("SAVED-C45-Out")
c45.id3.tree_.load_from_file("SAVED-C45-Out")
c45.id3.tree_.save_to_file("SAVED-2-C45-Out")
# if c45.prunedTree_ != []:
# 	print("-------------------------AFTER PRUNING--------------------------")
# 	c45.prunedTree_[0].export_tree()

# id3 = myID3(df, target, attributes)
# id3.tree_.export_tree()
# id3.tree_.load_from_file("Output-ID3")
# print(id3.tree_.data_from_file)
# print(id3.tree_.line_indexes_dict)
# id3.tree_.save_to_file("Output-ID3-Out")