#!/usr/bin/env python
from __future__ import unicode_literals, print_function

import ast
import pandas as pd
import nltk
pd.options.display.max_colwidth = 300
import glob, os
# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
Last tested with: v2.2.4
"""

import plac
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import en_core_web_sm

data_dir = "../data"

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)

def main(model=None, output_dir=None, n_iter=100, TRAIN_DATA=None):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    nlp = en_core_web_sm.load()

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    # for text, _ in TRAIN_DATA:
    #     doc = nlp(text)
    #     print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    #     print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # for text, _ in TRAIN_DATA:
        #     doc = nlp2(text)
        #     print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        #     print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

def outer_strip(word, char):
    if word[0] == word[-1] and word[0] == char:
        return word[1:-1]
    return word

if __name__ == "__main__":
    # plac.call(main)
    strings = []
    for fi in glob.glob(f"{data_dir}/training_data.csv"):
        df = None
        with open(fi) as csv_file:
            df = pd.read_csv(csv_file, header=0)
            df['text'] = df['text'].astype(str)
            df['organizations'] = df.apply((lambda x: ast.literal_eval(x['organizations'])), axis=1)
            for index, row in df.iterrows():
                orgs = row['organizations']
                orgs = [word.strip('][') for word in orgs if len(word) > 0]
                orgs = [word for word in orgs if len(word) > 0]
                if len(orgs) == 1 and orgs != [''] and orgs != ['"']:
                    if orgs[0] not in row['text']:
                        if not orgs[0].strip("'") in row['text']:
                            continue
                    start = str(row['text']).index(orgs[0])
                    end = start + len(orgs[0])
                    strings.append((row['text'][:start], row['text'][end:]))

    training = []
    with open(f"{data_dir}/Organizations.csv") as fi:
        df = pd.read_csv(fi, header=0)
        df = df.drop_duplicates(subset=['org_concept_name'])
        for index, row in df.iterrows():
            for comment in random.choices(strings, k=4):
                stri = str(comment[0] + row['org_concept_name'] + comment[1])
                org = str(row['org_concept_name'])
                training.append((stri, {"entities": [(len(comment[0]), len(comment[0]) + len(org), "ORG")]}))
    for fi in glob.glob(f"{data_dir}/training_data.csv"):
        df = None
        with open(fi) as csv_file:
            df = pd.read_csv(csv_file, header=0)
            df['text'] = df['text'].astype(str)
            df['organizations'] = df.apply((lambda x: ast.literal_eval(x['organizations'])), axis=1)
            for index, row in df.iterrows():
                orgs = row['organizations']
                orgs = [outer_strip(outer_strip(word.strip(']['), "'"), '"') for word in orgs if len(word) > 0]
                orgs = [word for word in orgs if len(word) > 0]
                if len(org) == 0 or org == [''] or org == ['"']:
                    training.append((str(row['text']), {"entities": []}))
    main(TRAIN_DATA=training, output_dir="spacy_ner_model")