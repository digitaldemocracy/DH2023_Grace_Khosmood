from glob import glob
from string import punctuation
from io import StringIO
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import numpy as np
pd.options.display.max_colwidth = 300
import os
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

vote_terms = set(["aye", "nay", "no", "yes", "yep", "nope"])
last_names_to_first = {}

data_dir = "../data/engagement_absentee_data/transcripts"

full_names = pd.read_csv(f"{data_dir}/full_people_list.csv")
discussion_names = pd.read_csv(f"{data_dir}/discussion_people_list.csv")

def find_votes(df, vote_counts):
    words = word_tokenize(df["text"].translate(str.maketrans('', '', punctuation)))
    all_names_set = set()
    last_names_dict = {}
    for item in vote_counts:
        all_names_set.add(item[0])
        all_names_set.add(item[1])
        last_names_dict[item[1].replace(" ", "").replace("-", "").replace("'", "").lower()] = item
    max_pos = len(words)
    for pos in range(len(words)):
        if str(words[pos]).lower() in last_names_dict:
            name = str(words[pos]).lower()
            while pos + 1 < max_pos and words[pos + 1] not in all_names_set:
                if words[pos + 1].lower() in vote_terms:
                    vote_counts[last_names_dict[name]] = 1
                    del last_names_dict[name]
                    break
                pos += 1
        elif pos + 1 < len(words) and str(words[pos] + words[pos+1]).lower() in last_names_dict:
            name = str(words[pos] + words[pos+1]).lower()
            while pos + 1 < max_pos and words[pos + 1] not in all_names_set:
                if words[pos + 1].lower() in vote_terms:
                    vote_counts[last_names_dict[name]] = 1
                    del last_names_dict[name]
                    break
                pos += 1

def convert_list_to_str(li):
    if len(li) == 0:
        return ""
    li_strs = []
    for name in li:
        li_strs.append(str(name).replace(",", ""))
    if len(li_strs) == 1:
        return li_strs[0]
    ret = ", ".join(li_strs[:-1])
    if len(li_strs) == 2:
        ret += " and " + li_strs[-1]
    else:
        ret += ", and " + li_strs[-1]
    return ret

def was_mentioned(df, legislators_dict):
    for name in legislators_dict:
        if name in df['text'] or (name in df['last'] and last_names_to_first[df['last']] in df['first']):
            legislators_dict[name] = True

vote_terms = set(["aye", "nay", "no", "yes", "yep", "nope"])
def find_num_comments(df, names):
    name_tup = (df['first'], df['last'])
    if name_tup in names:
        names[name_tup] += 1

def find_votes(df, vote_counts):
    words = word_tokenize(df["text"].translate(str.maketrans('', '', punctuation)))
    all_names_set = set()
    last_names_dict = {}
    for item in vote_counts:
        all_names_set.add(item[0])
        all_names_set.add(item[1])
        last_names_dict[item[1]] = item
    max_pos = len(words)
    for pos in range(len(words)):
        if words[pos] in last_names_dict:
            name = words[pos]
            while pos + 1 < max_pos and words[pos + 1] not in all_names_set:
                if words[pos + 1].lower() in vote_terms:
                    vote_counts[last_names_dict[name]] += 1
                    del last_names_dict[name]
                    break
                pos += 1

def get_name(pid):
    name = full_names[full_names['pid'] == pid].iloc[0]
    return name['first'], name['last']

def get_not_talking_legislators(df, did):
    legislator_counts = {}
    vote_counts = {}
    pids = set()
    # names of participants in the discussion
    names = get_names_participants(did)
    if names.empty or df is None:
        return None
    for _, person in names.iterrows():
        first, last = get_name(person['pid'])
        legislator_counts[(first, last)] = 0
        vote_counts[(first, last)] = 0
        pids.add(person['pid'])
    df[(df['last'] == 'Secretary') & (df['first'] == 'Committee')].apply(lambda x: find_votes(x, vote_counts), axis=1)
    df.apply(lambda x: find_num_comments(x, legislator_counts), axis=1)
    not_talking_legislators = []
    
    for key in legislator_counts:
        speaking_engagement = legislator_counts[key]
        vote_engagement = vote_counts[key]
        talking_score = speaking_engagement + vote_engagement
        name = str(key[0]) + ", " + str(key[1])
        if talking_score == 0:
            not_talking_legislators.append(name)
    return not_talking_legislators


def get_facts(df, non_talking_legislators):
    if non_talking_legislators == None or len(non_talking_legislators) == 0:
        return ([()], "")
    legislators_dict = {}
    for name in non_talking_legislators:
        first, last = name.split(", ")
        legislators_dict[last] = False
        last_names_to_first[last] = first
    df.apply(lambda x: was_mentioned(x, legislators_dict), axis=1)
    absentees = []
    num_absent = 0
    for name in legislators_dict:
        if not legislators_dict[name]:
            absentees.append(str(last_names_to_first[name] + ", " + name))
            num_absent += 1
    if num_absent / len(non_talking_legislators) >= .6 or len(absentees) == 0:
        return ([()], "")
    return ([tuple(absentees)], convert_list_to_str(absentees))

def get_names_participants(did):
    names = discussion_names[discussion_names['did'] == did].iloc[0]['people']
    if names.strip() == "":
        return pd.DataFrame()
    df = pd.read_csv(StringIO(names))
    df = df[df.apply(lambda x: "committee_member" in x['generated_pc'], axis=1)]
    return df

def run_query(dids):
    rows = []
    for did in dids:
        df = pd.read_csv(f"{data_dir}/discussion_{did}.csv")
        row, absentees = get_facts(df, get_not_talking_legislators(df, did))
        rows.append(row)
        if absentees is not None and absentees != '':
            print(f"In hearing {did}, {absentees} was absent")
    data = pd.DataFrame(rows)
    data.columns = ["absentees"]


transcripts = glob(f"{data_dir}/discussion_*.csv")
dids = []
RE_EXPR = r"discussion_(\d+).csv"
for path_name in transcripts:
    file_name = os.path.basename(path_name)
    match = re.match(RE_EXPR, file_name)
    if match is not None:
        dids.append(int(match.groups()[0]))

run_query(dids)