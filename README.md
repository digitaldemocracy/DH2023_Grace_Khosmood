#*Organization and Opinion Extraction in Public Comments and Legislator Engagement Tracking*

See also the arXiv version of this paper: https://arxiv.org/abs/2109.08855
This folder contains the training and testing methods for the paper *Organization and Opinion Extraction in Public Comments and Legislator Engagement Tracking*
This assumes that the data folder is in the same parent folder as code
In this repository, the "data_set.zip" file contains the content of "data" folder referenced below.

## Organization Affiliation Tracking

The organization tracker can be trained by running `affiliation_spacy_training.py` and `affiliation_stanford_ner_training.py`. This will generate the neccessary models in the code directory. To test the Stanford NER model individually, run `affiliation_stanford_ner_testing.py`, to only test the SpaCy NER model, run `affiliation_spacy_testing.py`. To test the full system, run `affiliation_testing.py`. The testing methods will print out any misidentifications from the model and the F1 error statistics. Please note that these incorrect identifications should be reviewed to correct any instances that were incorrectly marked as wrong.

## Position Tracking

To train and evaluate the position tracking model, run `position_test_train.py`. This uses the same data as the Organizational Affiliation Tracker. This will print any incorrect outputs from the model and calculate the final F1 score for the model. The script also pickles the model for later use.

## Engagement Tracking

The engagement tracker can be demonstrated by running `engagement_calculations.py`. This will run the tracker on all the hearing transcripts in data`/engagement_absentee_data`. This will calculate the engagement statistics for all legislative members listed in `data/engagement_absentee_data/full_people_list.csv`. Please note that for each hearing discussion transcript, information on the legislators on the committee for the hearing should be included in `data/engagement_absentee_data/discussion_people_list.csv`. 

Note that as only a small subset of the total data is included in the folder, the full engagement statistics can't be calculated. The script is included to demonstrate the methods used to generate the engagement list.

## Absentee Tracking

The absentee tracker can be run with `absentee_calculations.py`. This will run the tracker on all the hearing transcripts in `data/engagement_absentee_data`. This will calculate the absent legislative members from each hearing. Please note that for each hearing discussion transcript, information on the legislators on the committee for the hearing should be included in `data/engagement_absentee_data/discussion_people_list.csv`. This is neccessary because the script requires prior knowledge of who is expected to be present at the hearing.

Note that as only a small subset of the total data is included in the folder, the full absentee statistics can't be calculated. The script is included to demonstrate the methods used to generate the engagement list.
