
import pandas as pd
import numpy as np
pd.options.display.max_colwidth = 300
import glob, os
import spacy
import sklearn
from spacy import displacy
from pathlib import Path
from collections import Counter
import en_core_web_sm
import string
from sklearn.ensemble import RandomForestClassifier
import re
import pickle
import nltk

data_dir = "../data/"

strong_opposition_words = ["oppose", "opposition", "opposing", "opposed"]
strong_support_words = ["support", "supporting"]
medium_opposition_words = ["no vote", "nay vote"]
medium_support_words = ["aye vote", "yes vote"]
weak_support_words = ["co.?sponsor"]

true_positives = 0
false_positives = 0
false_negatives = 0

oppose = 0
neutral = 0
support = 0

train_data_val = []
train_data_true = []
train_data = []
test_data = []
test_data_true = []

def get_score(true_opinion, calculated_opinion, err_print):
    global true_positives
    global false_positives
    global false_negatives
    err = False
    if(not calculated_opinion in true_opinion and not calculated_opinion == true_opinion):
        if(calculated_opinion == "neutral"):
            false_negatives += 1
        else:
            false_positives += 1
        err = True
    else:
        true_positives += 1
    if err:
        print(err_print)

def find_stats(df):
    global support
    global neutral
    global oppose
    if(df['opinion'] == "support"):
        support += 1
    if(df['opinion'] == "oppose"):
        oppose += 1
    if(df['opinion'] == "neutral"):
        neutral += 1

def get_support(df):
    # Tokenize: Split sentence into words
    text = df["text"].lower()
    opinion = "neutral"
    polarity = 0
    for word in strong_support_words:
        polarity += len(re.findall(word, text))
    for word in strong_opposition_words:
        polarity -= len(re.findall(word, text))
    for word in medium_support_words:
        polarity += 0.75 * len(re.findall(word, text))
    for word in medium_opposition_words:
        polarity -= 0.75 * len(re.findall(word, text))
    for word in weak_support_words:
        polarity += 0.5 * len(re.findall(word, text))
    if(polarity >= .5):
        opinion = "support"
    if(polarity <= -.5):
        opinion = "oppose"
    err_str = "calculated: " +  str(opinion) + " | true: " + str(df["opinion"] + " | polarity: " + str(polarity))
    get_score(df["opinion"], opinion, err_str)

def generate_training_data(df):
    text = df["text"].lower()
    data = []
    for word in strong_support_words:
        data.append(len(re.findall(word, text)))
    for word in strong_opposition_words:
        data.append(len(re.findall(word, text)))
    for word in medium_support_words:
        data.append(len(re.findall(word, text)))
    for word in medium_opposition_words:
        data.append(len(re.findall(word, text)))
    for word in weak_support_words:
        data.append(len(re.findall(word, text)))
    train_data.append((data, df["opinion"]))
    train_data_true.append(df["opinion"])
    train_data_val.append(data)

def generate_testing_data(df):
    text = df["text"].lower()
    data = []
    for word in strong_support_words:
        data.append(len(re.findall(word, text)))
    for word in strong_opposition_words:
        data.append(len(re.findall(word, text)))
    for word in medium_support_words:
        data.append(len(re.findall(word, text)))
    for word in medium_opposition_words:
        data.append(len(re.findall(word, text)))
    for word in weak_support_words:
        data.append(len(re.findall(word, text)))
    test_data_true.append(df["opinion"])
    test_data.append(data)
    
def run_ml():
    for fi in glob.glob(f"{data_dir}/training_data.csv"):
        with open(fi) as csv_file:
            df = pd.read_csv(csv_file, header=0)
            if "opinion" not in df.columns:
                continue
            df = df[df['opinion'].notna()]
            df['text'] = df['text'].astype(str)
            mappings = {}
            for col in df.columns:
                mappings[col] = lambda column: max(column)
            mappings["text"] = lambda column: " ".join(column)
            mappings["organizations"] = lambda column: ",".join(column)
            df = df.groupby((df["pid"]!=df["pid"].shift()).cumsum()).agg(mappings, as_index=False).reset_index(drop=True)
            df = df.loc[df['person_type'] == "general public"]
            df.apply(generate_training_data, axis=1)
    for fi in glob.glob(f"{data_dir}/testing_data.csv"):
        with open(fi) as csv_file:
            df = pd.read_csv(csv_file, header=0)
            df = df[df['opinion'].notna()]
            df['text'] = df['text'].astype(str)
            mappings = {}
            for col in df.columns:
                mappings[col] = lambda column: max(column)
            mappings["text"] = lambda column: " ".join(column)
            mappings["organizations"] = lambda column: ",".join(column)
            df = df.loc[df['person_type'] == "general public"]
            df.apply(generate_testing_data, axis=1)
    classifier = sklearn.tree.DecisionTreeClassifier()
    matrix = np.matrix(train_data_val)
    classifier.fit(matrix, train_data_true)

    # uncomment to use generated model
    # with open('public_support_decision_tree.pickle', 'rb') as fi:
    #     classifier = pickle.load(fi) 

    matrix = np.matrix(test_data)
    out = classifier.predict(test_data)
    for val in range(len(out)):
        get_score(test_data_true[val], out[val], "OUT: " + str(out[val]) + " TRUE: " + str(test_data_true[val]))
    print("true positives: " + str(true_positives))
    print("false negatives: " + str(false_negatives))
    print("false positives: " + str(false_positives))
    precision = true_positives / (true_positives+false_positives)
    recall = true_positives / (true_positives+false_negatives)
    print("F1: " + str(2 * (precision*recall) / (precision + recall)))
    with open('public_support_decision_tree.pickle', 'wb') as fi:
        pickle.dump(classifier, fi)

run_ml()