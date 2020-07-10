import os
import json
import logging
import argparse

from tqdm import tqdm
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

from utils import set_console_logger

set_console_logger()
logger = logging.getLogger()
logging.getLogger(
    'gensim.utils').setLevel(logging.WARNING)



def load_corpus(dataset_dir):
    for month_dir in os.listdir(dataset_dir):
        month_path = os.path.join(dataset_dir, month_dir)
        if not os.path.isdir(month_path):
            continue
        for filename in os.listdir(month_path):
            path = os.path.join(month_path, filename)
            if path.endswith('.jsonl') and 'annotated' in path:
                with open(path) as f:
                    for line in f:
                        tweet = json.loads(line)
                        yield tweet


def main():
    logger.info(f'Loading data from {args.dataset_dir}')
    corpus = load_corpus(args.dataset_dir)
    model_path = os.path.join(args.dump_dir, 'lda.model')
    logger.info(f'Loading model from {model_path}')
    model = LdaModel.load(model_path)
    corpus_bow = (model.id2word.doc2bow(text['candidates']) for text in corpus)
    
    predictions_path = os.path.join(args.dump_dir, 'lda.prediction.jsonl')
    topic_ids = set()
    with open(predictions_path, 'w') as f:
        for tweet, tweet_bow in tqdm(zip(corpus, corpus_bow)):
            topics = model.get_document_topics(tweet_bow)
            topics = [(topic_id, topic_prob.item()) for topic_id, topic_prob in topics]
            tweet['topics'] = topics
            f.write(json.dumps(tweet) + '\n')
    logger.info(f'Predictions have been written to {predictions_path}')

    topics_path = os.path.join(args.dump_dir, 'lda.topics.txt')
    topics = model.show_topics(num_topics=model.num_topics,
                               num_words=10,
                               log=False,
                               formatted=True)

    with open(topics_path, 'w') as f:
        for topic_no, topic in topics:
            f.write(f'Topic {topic_no}: {topic}\n')
    logging.info(f'Topics have been written to {topics_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LDA Model')
    parser.add_argument('--dataset_dir', required=True,
                        help='dataset directory')
    parser.add_argument('--dump_dir', help='dump directory')
    args = parser.parse_args()
    if not args.dump_dir:
        args.dump_dir = os.path.join(args.dataset_dir, 'lda_dump')
    print(args)
    main()
