from typing import List
import json
import os
import re
import sys
import logging
import argparse
import gzip

from tqdm import tqdm
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from utils import set_console_logger

"""
Preprocessed.jsonl -> Annotated.jsonl
"""


set_console_logger()
logger = logging.getLogger()


COVID_KEYWORDS = {
    "coronavirus",
    "coronavirusoutbreak",
    "coronavirusoutbreak",
    "covid",
    "coronavirus",
    "covid19",
    "covid2019",
    "covid_19",
    "covid-19",
    "covidー19",
    "virus",
    "corona",
    "don",
    "amp",
    "didn",
    "dont",
    "isn"
}


def process(tokenizer, lemmatizer, text):
    # https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py
    tokens = tokenizer.tokenize(text)
    candidates = []
    candidates_idxs = []
    for token_idx, token in enumerate(tokens):
        token = token.lower()
        if token == 'url' or token == 'user_mention':
            continue
        elif token.isnumeric():
            continue
        elif token in STOP_WORDS:
            continue
        elif token in COVID_KEYWORDS:
            continue
        if len(token) < 3:
            continue
        token = lemmatizer.lemmatize(token)
        candidates.append(token)
        candidates_idxs.append(token_idx)

    return tokens, candidates, candidates_idxs


def extract_candidate(tokenizer, lemmatizer, data_path):
    basename = os.path.basename(data_path)
    path = os.path.dirname(data_path)
    output_file = basename.replace(
        'preprocessed', 'annotated').replace('.gz', '')
    output_path = os.path.join(path, output_file)

    with gzip.open(data_path, 'rt') as f, open(output_path, 'w') as out_f:
        for idx, line in enumerate(f):
            tweet = json.loads(line)
            full_text = tweet['preprocessed_full_text']

            tokens, candidates, candidates_idxs = process(tokenizer, lemmatizer, full_text)

            tweet['tokens'] = tokens
            tweet['candidates'] = candidates
            tweet['candidates_idxs'] = candidates_idxs

            out_f.write(json.dumps(tweet) + '\n')


def find_paths(dataset_dir):
    data_files = []
    for month_dir in os.listdir(dataset_dir):
        month_path = os.path.join(dataset_dir, month_dir)
        if not os.path.isdir(month_path):
            continue
        for filename in os.listdir(month_path):
            data_file = os.path.join(month_path, filename)
            if re.match(r'coronavirus-tweet-preprocessed-2020-\d\d-\d\d-\d\d.jsonl.gz', filename):
                if args.force:
                    data_files.append(data_file)
                    continue

                output_file = filename.replace(
                    'preprocessed', 'annotated').replace('.gz', '')
                output_path = os.path.join(month_path, output_file)

                if os.path.isfile(output_path):
                    with open(output_path) as f:
                        anno_tweets = 0
                        for line in f:
                            anno_tweets += 1
                    with gzip.open(data_file, 'rt') as f:
                        origin_tweets = 0
                        for line in f:
                            origin_tweets += 1
                    if anno_tweets == origin_tweets:
                        logger.info(
                            f'Annotation complete for {data_file}. Skip.')
                    else:
                        logger.warning(
                            f'Annotation not complete for {data_file}. Overwriting.')
                        data_files.append(data_file)
                else:
                    data_files.append(data_file)
    return data_files


def main():
    data_files = find_paths(args.dataset_dir)
    logger.info(f'{len(data_files)} data files to be annotated.')
    if len(data_files) == 0:
        return

    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()

    for data_file in tqdm(data_files, total=len(data_files)):
        extract_candidate(tokenizer, lemmatizer, data_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract candidates from tweets')
    parser.add_argument('--dataset_dir', required=True,
        help='dataset directory')
    parser.add_argument('--force', '-f', action='store_true',
                        help='if processed file exists, overwrite.')

    args = parser.parse_args()
    print(args)
    main()
