import ast
import pandas as pd
pd.options.display.max_colwidth = 300
import glob, os
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import string
from pathlib import Path
os.chdir(".")
opposition_words = set(["oppose", "opposition", "opposing"])
support_words = set(["support", "supporting"])
linking_words = set(["of", "in", ","])
signal_words = set(["with", "from", "behalf"])
true_positives = 0
false_positives = 0
false_negatives = 0
output_dir = Path("spacy_ner_model")
nlp = spacy.load(output_dir)

data_dir = "../data"

# https://stackoverflow.com/questions/16891340/remove-a-prefix-from-a-string
def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]

def get_score(organizations, found):
    global true_positives
    global false_positives
    global false_negatives
    num_false_positives = 0
    errs = 0
    found = set([word.lower().translate(str.maketrans('', '', string.punctuation)) for word in found])
    organizations = set([word.lower().translate(str.maketrans('', '', string.punctuation)) for word in organizations if word.lower().translate(str.maketrans('', '', string.punctuation)) != ""])
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
        print(found)
        print(organizations)
        print(old_orgs)
        print("")

def find_orgs(df):
    text = df["text"]
    doc = nlp(text)
    entities = []
    prev = None
    for entity in doc.ents:
        if entity.label_ == "PERSON":
            continue
        if(entity.label_ == "ORG"):
            entities.append(str(doc[entity.start : entity.end]))
            
    if prev is not None:
        if prev.label_ != "GPE":
            entities.append(str(prev.text))
    get_score(df["organizations"], entities)

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