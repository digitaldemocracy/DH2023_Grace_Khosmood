#!/usr/bin/env python

import pandas as pd
pd.options.display.max_colwidth = 300
import glob, os
import ast
import plac
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import en_core_web_sm
import re
import subprocess
import nltk

data_dir = "../data"

def outer_strip(word, char):
    if word[0] == word[-1] and word[0] == char:
        return word[1:-1]
    return word

if __name__ == "__main__":
    orgs = []
    with open(f"{data_dir}/Organizations.csv") as fi:
        df = pd.read_csv(fi, header=0)
        df = df.drop_duplicates(subset=['org_concept_name'])
        for index, row in df.iterrows():
            orgs.append(str(row['org_concept_name']))
    nlp = spacy.load("en_core_web_sm")
    with open("stanford_ner_training_data.tsv", 'w') as out_file:
        for fi in glob.glob(f"{data_dir}/training_data.csv"):
            df = None
            with open(fi) as csv_file:
                df = pd.read_csv(csv_file, header=0)
                df['text'] = df['text'].astype(str)
                df['organizations'] = df.apply((lambda x: ast.literal_eval(x['organizations'])), axis=1)
                for index, row in df.iterrows():
                    organizations = row['organizations']
                    orgs = [outer_strip(outer_strip(word.strip(']['), "'"), '"') for word in orgs if len(word) > 0]
                    organizations = [word for word in organizations if len(word) > 0]
                    split_str = "|".join(organizations)
                    sent = row["text"]
                    if len(split_str) > 0:
                        reassemble = re.split(split_str, sent)
                    else:
                        reassemble = [sent]
                    for i in range(100):
                        string = ""
                        for substr in reassemble[:-1]:
                            for word in nltk.word_tokenize(substr):
                                if word in row['first'] or word in row['last']:
                                    out_file.write(word + "\tPERSON\n")
                                else:
                                    out_file.write(word + "\tO\n")
                            for word in nltk.word_tokenize(random.choice(orgs)):
                                out_file.write(word + "\tORGANIZATION\n")
                        for word in nltk.word_tokenize(reassemble[-1]):
                            out_file.write(word + "\tO\n")
        for organizations in orgs:
            out_file.write(str(organizations) + "\tORGANIZATION\n")
        for i in range(1000):
            out_file.write("Uber\tORGANIZATION\n")
cmd = f"java -Xmx12g -cp {data_dir}/stanford-ner-4.2.0.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop training_props.txt"
cmd_arr = cmd.split(" ")
subprocess.run(cmd_arr)