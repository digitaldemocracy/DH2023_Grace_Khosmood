import os
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import pandas as pd
from glob import glob
from string import punctuation
from io import StringIO

data_dir = "../data/engagement_absentee_data/transcripts"

full_names = pd.read_csv(f"{data_dir}/full_people_list.csv")
discussion_names = pd.read_csv(f"{data_dir}/discussion_people_list.csv")

def find_num_comments(df, names):
    name_tup = (df['first'], df['last'])
    if name_tup in names and df['text'] is not None and len(df['text']) > 6:
        names[name_tup] += 1


vote_terms = set(["aye", "nay", "no", "yes", "yep", "nope"])


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

def get_names_participants(did):
    names = discussion_names[discussion_names['did'] == did].iloc[0]['people']
    if names.strip() == "":
        return pd.DataFrame()
    df = pd.read_csv(StringIO(names))
    df = df[df.apply(lambda x: "committee_member" in x['generated_pc'], axis=1)]
    return df

def find_num_question_marks(df, names):
    name_tup = (df['first'], df['last'])
    if name_tup in names:
        if df['text'] == None or df['text'].count("?") == 0:
            return
        sentences = sent_tokenize(df['text'])
        for sentence in sentences:
            if sentence.count("?") > 0 and len(word_tokenize(sentence)) > 8:
                names[name_tup] += 1


def combine_speech(df):
    return df.groupby((df["pid"] != df["pid"].shift()).cumsum()).agg(
        text=('text', lambda column: " ".join(column)), pid=('pid', "max"), first=('first', "max"), last=('last', "max")
    ).reset_index(drop=True)


def get_AxA(df, names):
    AxA_df = combine_speech(df)
    if len(AxA_df) == 0:
        return
    for ind in range(0, AxA_df.index.stop - 2):
        if AxA_df.iloc[ind]['pid'] == AxA_df.iloc[ind+2]['pid']:
            name = (AxA_df.iloc[ind]['first'], AxA_df.iloc[ind]['last'])
            if name in names and (AxA_df.iloc[ind+1]['first'], AxA_df.iloc[ind+1]['last']) not in names:
                text = AxA_df.iloc[ind]['text'] + " " + AxA_df.iloc[ind+2]['text']
                words = word_tokenize(text)
                num_words = len(words)
                names[name].append(num_words)

def get_name(pid):
    name = full_names[full_names['pid'] == pid].iloc[0]
    return name['first'], name['last']

def get_facts(did):
    df = pd.read_csv(f"{data_dir}/discussion_{did}.csv")
    speaking_counts = {}
    question_counts = {}
    vote_counts = {}
    AxA_counts = {}
    pids = set()
    names = get_names_participants(did)
    if names.empty or df is None:
        return None
    for _, person in names.iterrows():
        first, last = get_name(person['pid'])
        speaking_counts[(first, last)] = 0
        vote_counts[(first, last)] = 0
        question_counts[(first, last)] = 0
        AxA_counts[(first, last)] = []
        pids.add(person['pid'])
    df[(df['last'] == 'Secretary') & (df['first'] == 'Committee')].apply(lambda x: find_votes(x, vote_counts),
                                                                         axis=1)
    df.apply(lambda x: find_num_comments(x, speaking_counts), axis=1)
    df.apply(lambda x: find_num_question_marks(x, question_counts), axis=1)
    get_AxA(df, AxA_counts)
    return [did, speaking_counts, vote_counts, question_counts, AxA_counts]

from collections import defaultdict
def def_value():
    # num votes, num hearings on committee, num times speaking, words in back and forth, num questions
    return [0, 0, 0, 0, 0]
legislators_stats = defaultdict(def_value)
engagement_scores = {}

def filter_by_length(length):
    return length > 12

def process_hearing(row, legislators_stats):
    for legislator in row['speaking_counts']:
        # number of times votes
        legislators_stats[legislator][0] += row['vote_counts'][legislator]
        # number of opportunites is vote
        legislators_stats[legislator][1] += 1
        # number of speaking blocks
        legislators_stats[legislator][2] += row['speaking_counts'][legislator]
        # total length of AOA
        legislators_stats[legislator][3] += sum(filter(filter_by_length, row['AxA_counts'][legislator]))
        # number of comments
        legislators_stats[legislator][4] += row['question_counts'][legislator]
    pass

alpha = 0.5
beta = 0.0005
gamma = 0.00005
delta = 0.01

def compute_hearing_engagement(legislators_stats, engagement_scores):
    for legislator in legislators_stats:
        legislator_stat = legislators_stats[legislator]
        vote_score = alpha * legislator_stat[0] / legislator_stat[1] 
        speaking_score = beta * legislator_stat[2]
        AoA_score = gamma * legislator_stat[3]
        question_score = delta * legislator_stat[4]
        engagement_scores[legislator] = [vote_score + speaking_score + AoA_score + question_score, vote_score, speaking_score, AoA_score, question_score]

def compute_engagement(df):
    legislators_stats = defaultdict(def_value)
    engagement_scores = {}
    df.apply(lambda x: process_hearing(x, legislators_stats), axis=1)
    compute_hearing_engagement(legislators_stats, engagement_scores)
    return engagement_scores

def print_engagement_scores(engagement_scores):
    engagement_counts = sorted(engagement_scores.items(), key=lambda x: x[1][0])
    formatted_engagement_counts = []
    num = 1
    for legislator in engagement_counts[::-1]:
        formatted_engagement_counts.append((f"{num}", f"{legislator[0][0]} {legislator[0][1]}:", \
                f"Overall Score: {round(legislator[1][0], 3)}", 
                f"Vote Score: {round(legislator[1][1], 6)}", 
                f"Speaking Score: {round(legislator[1][2], 3)}", 
                f"AoA Score: {round(legislator[1][3], 3)}", 
                f"Question Score: {round(legislator[1][4], 3)}"))
        num+=1

    lens = []
    for col in zip(*formatted_engagement_counts):
        lens.append(max([len(v) for v in col]))
    format = "   ".join(["{:<" + str(l) + "}" for l in lens])
    for row in formatted_engagement_counts:
        print(format.format(*row))

def run_query(dids):
    rows = []
    for did in dids:
        row = get_facts(did)
        if row is None:
            continue
        rows.append(row)

    data = pd.DataFrame(rows)
    data.columns = ["did", "speaking_counts", "vote_counts", "question_counts", "AxA_counts"]
    print_engagement_scores(compute_engagement(data))


transcripts = glob(f"{data_dir}/discussion_*.csv")
dids = []
RE_EXPR = r"discussion_(\d+).csv"
for path_name in transcripts:
    file_name = os.path.basename(path_name)
    match = re.match(RE_EXPR, file_name)
    if match is not None:
        dids.append(int(match.groups()[0]))

run_query(dids)
