import pandas as pd
from myc45 import *

pd.options.mode.chained_assignment = None

def classify_id3(id3):
    dict = {
        "sepal_length": 5.5,
        "sepal_width": 3.4,
        "petal_length": 1.4,
        "petal_width": 0.6
    }
    isFloat = True
    print(id3.classify(id3.tree_, dict, isFloat))

def classify_c45(c45):
    dict = {
        "sepal_length": 5.5,
        "sepal_width": 3.3,
        "petal_length": 2.0,
        "petal_width": 0.65
    }
    print(c45.classify(c45.id3.tree_, dict))

def start_id3_experiment(id3):
    id3.tree_.save_to_file("Output-ID3.txt")
    id3.tree_.load_from_file("Output-ID3.txt")
    classify_id3(id3)

def start_c45_experiment(c45):
    c45.id3.tree_.save_to_file("Output-C45.txt")
    c45.id3.tree_.load_from_file("Output-C45.txt")
    classify_c45(c45)

def main_c45():
    c45 = myC45(df, target, attributes)
    start_c45_experiment(c45)

def main_id3():
    id3 = myID3(df, target, attributes)
    start_id3_experiment(id3)

df = pd.read_csv('iris.csv', sep=',')
attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target = 'species'

# Main Program Goes Here:
# main_id3()
main_c45()