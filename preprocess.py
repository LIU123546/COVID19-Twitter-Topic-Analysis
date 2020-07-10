import os
import re
import json
import gzip
import argparse
import logging
from multiprocessing import Pool as ProcessPool

from tqdm import tqdm
from emoji import demojize

from utils import set_console_logger


"""
 -> Preprocessed.jsonl
"""

set_console_logger()
logger = logging.getLogger()

# Preprocessing scripts:
# https://github.com/abdulfatir/twitter-sentiment-analysis/blob/master/code/preprocess.py
# https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py


def preprocess_word(word):
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    return word


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def preprocess_tweet(tweet):
    # Replaces URLs with special token
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    # Replace @handle with special token
    tweet = re.sub(r'@[\S]+', ' USER_MENTION ', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    # Replace special symbols.
    tweet = tweet.replace("’", "'").replace("…", "...")
    # normalize am / pm
    tweet = tweet.replace(" p . m .", "  p.m.") .replace(" p . m ", " p.m ").replace(" a . m .", " a.m.").replace(" a . m ", " a.m ")
    # normalize 1/2 multiple tweets
    tweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", tweet)
    tweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", tweet)
    tweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", tweet)
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace emojis with emoji.emojize
    tweet = demojize(tweet)

    tweet_tokens = [preprocess_word(word) for word in tweet.split()]

    return ' '.join(tweet_tokens)


def find_paths(dataset_dir):
    data_files = []
    for month_dir in os.listdir(dataset_dir):
        month_path = os.path.join(dataset_dir, month_dir)
        if not os.path.isdir(month_path):
            continue
        for filename in os.listdir(month_path):
            path = os.path.join(month_path, filename)
            if re.match(r'coronavirus-tweet-2020-\d\d-\d\d-\d\d.jsonl', filename):
                outputfile = filename.replace('coronavirus-tweet-', 'coronavirus-tweet-preprocessed-')
                output_path = os.path.join(args.output_dir, month_dir, outputfile)
                if args.force:
                    data_files.append((path, output_path))
                    continue
                if os.path.isfile(output_path):
                    with open(output_path) as f:
                        preprocessed_tweets = 0
                        for line in f:
                            preprocessed_tweets += 1
                    with open(path) as f:
                        origin_tweets = 0
                        for line in f:
                            origin_tweets += 1
                    if preprocessed_tweets == origin_tweets:
                        logger.info(
                            f'Preprocessing complete for {path}. Skip.')
                    else:
                        logger.warning(
                            f'Preprocessing not complete for {path}. Overwriting.')
                        data_files.append((path, output_path))
                else:
                    data_files.append((path, output_path))
    return data_files


def process_tweets_file(data_file):
    input_path, output_path = data_file
    date = os.path.basename(input_path)[len('coronavirus-tweet-'):-len('.jsonl')-3]
    f = open(input_path)
    if args.compress:
        output_path += '.gz'
    out_f = gzip.open(output_path, 'wt') if args.compress else open(output_path, 'w')
    for line in f:
        tweet = json.loads(line)
        tweet_id = tweet['id_str']
        created_at = tweet['created_at']
        full_text = tweet['full_text']
        processed = preprocess_tweet(full_text)
        processed_tweet = {
            'created_at': created_at,
            'date': date,
            'id_str': tweet_id,
            'full_text': full_text,
            'preprocessed_full_text': processed,
            'country': tweet['place']['country']
        }
        out_f.write(json.dumps(processed_tweet) + '\n')
    f.close()
    out_f.close()


def main():
    data_files = find_paths(args.input_dir)
    logger.info(f'{len(data_files)} data files found.')
    if len(data_files) == 0:
        return
    workers = ProcessPool(args.num_workers)
    with tqdm(total=len(data_files)) as pbar:
        for _ in tqdm(workers.imap_unordered(process_tweets_file, data_files)):
            pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Proprocessing Tweets')
    parser.add_argument('--input_dir', required=True, help='dataset directory')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of CPU processes')
    parser.add_argument('--compress', action='store_true', help='compress with gzip')
    parser.add_argument('--force', '-f', action='store_true', help='force overwriting existing files')
    args = parser.parse_args()
    print(args)
    main()
