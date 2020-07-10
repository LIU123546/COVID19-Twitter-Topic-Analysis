#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2020-04-30 22:54:00
# @Last Modified by: Shuailong
# @Last Modified time: 2020-04-30 22:54:16

import re
import os
import json
import random
import argparse
import gzip
from multiprocessing import Pool as ProcessPool
from collections import Counter

from tqdm import tqdm

"""
Explore data downloaded from https://github.com/echen102/COVID-19-TweetIDs

Filter original tweets with country information

Original keys:
['created_at', 'id', 'id_str', 'full_text', 'truncated', 'display_text_range', 'entities', 'source', 'in_reply_to_status_id', 'in_reply_to_status_id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place', 'contributors', 'retweeted_status', 'is_quote_status', 'retweet_count', 'favorite_count', 'favorited', 'retweeted', 'lang']

Filtered keys:
None:
    [in_reply_to_status_id]
Not None:
    [id, full_text, create_at and country, country_code, lang]
"""


def read_gz(filename):
    try:
        with gzip.open(filename) as f:
            for line in f:
                tweet = json.loads(line)
                yield tweet
    except Exception as e:
        print(e)
        print(f'Errors reading {filename}, skip')
        yield None


def filter_base(tweet):
    return tweet is not None


def filter_by_lang(tweet):
    return tweet['lang'] == 'en'


def filter_by_geo(tweet):
    return tweet['place'] and 'country' in tweet['place']


def filter_by_popularity(tweet):
    return tweet['retweet_count'] > 200 or tweet['favorite_count'] > 50


def filter_tweet(file_path):
    basename = os.path.basename(file_path)
    hour = basename[len('coronavirus-tweet-id-'):-len('.jsonl.gz')]
    date = hour[:-3]
    month = date[:-3]
    hour = hour[-2:]
    output_dir = os.path.join(args.output_dir, month)
    outpath = os.path.join(output_dir, f'coronavirus-tweet-{date}-{hour}.jsonl')
    
    filters = [filter_name_to_func[filter_] for filter_ in args.filters]
    tweets = read_gz(file_path)
    filtered_tweets = filter(lambda x: all(f(x) for f in filters), tweets)

    with open(outpath, 'w') as wf:
        for tweet in filtered_tweets:
            wf.write(json.dumps(tweet) + '\n')


def main():
    data_files = []
    for data_dir in args.input_dirs:
        for filename in os.listdir(data_dir):
            path = os.path.join(data_dir, filename)
            if filename.endswith('.jsonl.gz'):
                data_files.append(path)

    print(f'{len(data_files)} data files found.')

    workers = ProcessPool(args.num_workers)
    with tqdm(total=len(data_files)) as pbar:
        for _ in tqdm(workers.imap_unordered(filter_tweet, data_files)):
            pbar.update()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect filters')
    parser.add_argument('--input_dirs', '-i', type=str,
                        nargs='+', help='input dirs')
    parser.add_argument('--output_dir',  '-o', type=str, help='output file')
    parser.add_argument('--filters', nargs='+', default=['lang', 'geo'], choices=['lang', 'geo', 'popularity'])
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of CPU processes')
    args = parser.parse_args()
    print(args)

    args.filters.insert(0, 'base')

    filter_name_to_func = {
        'base': filter_base,
        'lang': filter_by_lang,
        'geo': filter_by_geo,
        'popularity': filter_by_popularity
    }

    main()


