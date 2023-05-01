import ast
import pandas as pd
pd.options.display.max_colwidth = 300
import glob, os
import spacy
from spacy import displacy
from pathlib import Path
from collections import Counter
import en_core_web_sm
import string
import re

import nltk
from nltk.tag.stanford import StanfordNERTagger

data_dir = "../data"
jar = f'{data_dir}/stanford-ner-4.2.0.jar'
model = './stanford-ner-model.ser.gz'

# Prepare NER tagger with english model
ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')

true_positives = 0
false_positives = 0
false_negatives = 0

def get_score(organizations, found, err_print):
    global true_positives
    global false_positives
    global false_negatives
    num_false_positives = 0
    errs = 0
    found = set([word.lower().translate(str.maketrans('', '', string.punctuation)) for word in found])
    organizations = set([word.lower().translate(str.maketrans('', '', string.punctuation)) for word in organizations])
    old_orgs = organizations.copy()
    for value in found:
        if value in organizations:
            true_positives += 1
            organizations.remove(value)
        else:
            num_false_positives += 1
            errs+=1
    new_found = set()
    if len(organizations) > 0:
        for string_found in found:
            for value in organizations:
                words = remove_prefix(value, "the ")
                words2 = remove_prefix(string_found, "the ")
                if(len(words) < 1 and len(words2) < 1):
                    continue
                if words == string_found or words2 == value or string_found == value:
                    if num_false_positives > 0:
                        num_false_positives -= 1
                    errs -= 1
                    organizations.remove(value)
                    break
    for val in found:
        for value in val.split(", "):
            for value2 in value.split(" and "):
                new_found.add(remove_prefix(value2, "the "))
    if len(organizations) > 0:
        for string_found in new_found:
            for value in organizations:
                words = remove_prefix(value, "the ")
                words2 = remove_prefix(string_found, "the ")
                if(len(words) < 1 and len(words2) < 1):
                    continue
                if words == string_found or words2 == value or string_found == value:
                    if num_false_positives > 0:
                        num_false_positives -= 1
                    errs -= 1
                    organizations.remove(value)
                    break
    new_found = set([word.replace(",", "") for word in found])
    for value in new_found:
        if value in organizations:
            true_positives += 1
            num_false_positives -= 1
            errs-=1
            organizations.remove(value)

    false_negatives += len([word for word in organizations if word != '' and word !='"'])
    errs += len([word for word in organizations if word != '' and word !='"'])
    false_positives += num_false_positives
    if errs > 0:
        print(err_print)
        print(found)
        print(organizations)
        print(old_orgs)
        print("")

# https://stackoverflow.com/questions/16891340/remove-a-prefix-from-a-string
def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]

def find_orgs(df):
    # Tokenize: Split sentence into words
    words = nltk.word_tokenize(df["text"])

    words = ner_tagger.tag(words)
    buffer = ""
    output_set = set()
    pos = 0
    length = 0

    for word in range(len(words)):
        if words[word][1] != "O" and words[word][1] != "PERSON":
            if(words[word][0][0] in ".,:;'\""):
                buffer = buffer[:-1]
            buffer += words[word][0] + " "
            length += 1
        elif buffer != "":
            if word + 1 < len(words):
                if(words[word][0] == "and" and words[word+1][1] != "O" and words[word+1][1] != "PERSON"):
                    buffer += words[word][0] + " "
                    length += 1
                    continue
                if word + 2 < len(words):
                    if(words[word][0] == "and" and words[word+1][0] == "the" and words[word+2][1] != "O" and words[word+2][1] != "PERSON"):
                        buffer += words[word][0] + " "
                        length += 1
                        continue
                if(words[word][0] == "the" and words[word+1][1] != "O" and words[word+1][1] != "PERSON"):
                    buffer += words[word][0] + " "
                    length += 1
                    continue
            
            buffer = remove_prefix(buffer, "the ").strip(". ,'")
            if buffer not in output_set:
                output_set.add(buffer)
            buffer = ""
            pos -= length
            length = 0
            continue
        pos += 1
   
    ents2 = set()
    for key in output_set:
        key_trans = key.lower().translate(str.maketrans('', '', string.punctuation))
        if key_trans in known_counties or key_trans in known_cities:
            continue
        if re.search("^board.*member$", key_trans) is not None: 
            continue
        if len(key_trans) < 3:
            continue
        ents2.add(key)
    ents = ents2
    ents2 = set()
    for elem in ents:
        words = nltk.word_tokenize(elem)
        total = elem
        for word in words[::-1]:
            if(word.islower()):
                total = total[:total.rindex(word)].strip(" ")
            else:
                ents2.add(total)
                break
        if(total == ""):
            ents2.add(elem)

    ents = ents2
    li = df["organizations"]
    li = [word for word in li if len(word) > 0]
    err_str = "ents: " + str(ents) + "\nstring: " + str(df['text'])
    get_score(li, ents, err_str)


def get_support(df):
    find_orgs(df)

num = 0
known_cities = set()
known_counties = set()
known_orgs = set()
known_orgs_dict = {}
with open(f"{data_dir}/Organizations.csv") as fi:
    df = pd.read_csv(fi, header=0)
    df = df.drop_duplicates(subset=['org_concept_name'])
    for index, row in df.iterrows():
        known_orgs.add(row['org_concept_name'].lower().translate(str.maketrans('', '', string.punctuation)))
        known_orgs_dict[row['org_concept_name'].lower().translate(str.maketrans('', '', string.punctuation))] = row['org_concept_name']

with open(f"{data_dir}/california_cities.csv") as fi:
    df = pd.read_csv(fi, header=0)
    df = df.drop_duplicates(subset=['City'])
    for index, row in df.iterrows():
        known_cities.add(row['City'].lower().translate(str.maketrans('', '', string.punctuation)))
        known_cities.add(row['City'].lower().translate(str.maketrans('', '', string.punctuation)) + " california")
    known_cities.add("california")
    df = df.drop_duplicates(subset=['County'])
    for index, row in df.iterrows():
        known_cities.add(str(row['County'].lower().translate(str.maketrans('', '', string.punctuation))) + " county")

num = 0
for fi in glob.glob(f"{data_dir}/testing_data.csv"):
    df = None
    print(fi)
    with open(fi) as csv_file:
        df = pd.read_csv(csv_file, header=0)
        df['text'] = df['text'].astype(str)
        df['organizations'] = df.apply((lambda x: ast.literal_eval(x['organizations'])), axis=1)
        df.apply(find_orgs, axis=1)
        num += len(df.index)

print("num: " + str(num))
print("true positives: " + str(true_positives))
print("false negatives: " + str(false_negatives))
print("false positives: " + str(false_positives))

precision = true_positives / (true_positives+false_positives)
recall = true_positives / (true_positives+false_negatives)
print("F1: " + str(2 * (precision*recall) / (precision + recall)))
