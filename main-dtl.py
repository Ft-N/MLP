import pandas as pd
from myc45 import *

pd.options.mode.chained_assignment = None

def classify_c45(c45):
    dict = {
        "sepal_length": 5.5,
        "sepal_width": 3.3,
        "petal_length": 2.0,
        "petal_width": 0.65
    }
    print(c45.classify(c45.id3.tree_, dict))

def start_c45_experiment(c45):
    c45.id3.tree_.save_to_file("Output-C45.txt")
    c45.id3.tree_.load_from_file("Output-C45.txt")
    classify_c45(c45)

def main_c45():
    c45 = myC45(df, target, attributes)
    start_c45_experiment(c45)

df = pd.read_csv('iris.csv', sep=',')
attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target = 'species'

# Main Program Goes Here:
main_c45()