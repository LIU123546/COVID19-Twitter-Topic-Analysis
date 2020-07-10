import os
import sys
import json
import logging
import argparse

import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from gensim.models.callbacks import PerplexityMetric, CoherenceMetric
from gensim.models.wrappers import LdaMallet
from gensim.models import LdaMulticore as GensimLdaMulticore
from gensim.models import LdaModel as GensimLdaModel
from tqdm import tqdm
from pprint import pprint

from ldamulticore import LdaModel, LdaMulticore
from utils import set_tee_logger


logging.getLogger(
    'gensim.topic_coherence').setLevel(logging.WARNING)
logging.getLogger(
    'gensim.utils').setLevel(logging.WARNING)


TOKEN_MIN_DOCS = 5
TOKEN_MAX_DOCS_FRAC = 0.5
NGRAM_MIN_FREQ = 5


def load_corpus(dataset_dir):
    corpus = []
    for month_dir in os.listdir(dataset_dir):
        month_path = os.path.join(dataset_dir, month_dir)
        if not os.path.isdir(month_path):
            continue
        data_files = os.listdir(month_path)
        for filename in tqdm(data_files, total=len(data_files), desc=month_dir):
            path = os.path.join(month_path, filename)
            if path.endswith('.jsonl') and 'annotated' in path:
                with open(path) as f:
                    for line in f:
                        tweet = json.loads(line)
                        corpus.append(tweet['candidates'])
    return corpus


def main():
    logger.info('-'*80)
    logger.info('Loading data')
    corpus = load_corpus(args.dataset_dir)

    logger.info('-'*80)
    logger.info('Make dictionary')

    dictionary = Dictionary(corpus)
    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=TOKEN_MIN_DOCS, no_above=TOKEN_MAX_DOCS_FRAC)

    vocab_path = os.path.join(args.dump_dir, 'vocab.txt')
    with open(vocab_path, 'w') as f:
        f.write("\n".join(dictionary.itervalues()) + '\n')

    # Bag-of-words representation of the documents.
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]

    logger.info(f'Number of unique tokens: {len(dictionary)}')
    logger.info(f'Number of documents: {len(bow_corpus)}')

    logger.info('-'*80)
    logger.info('Training model')

    callbacks = []
    if 'perplexity' in args.callbacks:
        perplexity_metric = PerplexityMetric(corpus=bow_corpus)
        callbacks.append(perplexity_metric)
    if 'coherence' in args.callbacks:
        coherence_metric = CoherenceMetric(texts=corpus,
                                        dictionary=dictionary,
                                        coherence=args.coherence,
                                        topn=args.topn)
        callbacks.append(coherence_metric)

    model_path = os.path.join(args.dump_dir, 'lda.model')
    if args.model == 'lda':
        model = LdaModel(corpus=bow_corpus,
                        num_topics=args.num_topics,
                        id2word=dictionary,
                        passes=args.num_epochs,
                        update_every=1,
                        eval_every=args.eval_every,
                        iterations=args.iterations,
                        alpha='auto',
                        eta='auto',
                        chunksize=args.batch_size,
                        callbacks=callbacks,
                        log_dir=args.log_dir,
                        model_dir=model_path
                        )
    elif args.model == 'multicore_lda':
        model = LdaMulticore(corpus=bow_corpus,
                            num_topics=args.num_topics,
                            id2word=dictionary,
                            passes=args.num_epochs,
                            eval_every=args.eval_every,
                            iterations=args.iterations,
                            eta='auto',
                            chunksize=args.batch_size,
                            workers=args.workers,
                            callbacks=callbacks,
                            log_dir=args.log_dir,
                            model_dir=model_path
                            )
    elif args.model == 'mallet_lda':
        model = LdaMallet(args.mallet_path,
                          corpus=bow_corpus,
                          num_topics=args.num_topics,
                          id2word=dictionary,
                          workers=args.workers,
                          prefix= os.path.join(args.dump_dir, 'mallet_'),
                          iterations=args.iterations)
    elif args.model == 'gensim_lda':
        model = GensimLdaModel(corpus=bow_corpus,
                         num_topics=args.num_topics,
                         id2word=dictionary,
                         passes=args.num_epochs,
                         update_every=1,
                         eval_every=args.eval_every,
                         iterations=args.iterations,
                         alpha='auto',
                         eta='auto',
                         chunksize=args.batch_size
                         )
    elif args.model == 'gensim_multicore_lda':
        model = GensimLdaMulticore(corpus=bow_corpus,
                                    num_topics=args.num_topics,
                                    id2word=dictionary,
                                    passes=args.num_epochs,
                                    eval_every=args.eval_every,
                                    iterations=args.iterations,
                                    eta='auto',
                                    chunksize=args.batch_size,
                                    workers=args.workers
                                    )

    model.save(model_path)

    logger.info('-'*80)

    if args.model != 'mallet_lda':
        top_topics = model.top_topics(texts=corpus, coherence='c_v')
        # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
        avg_topic_coherence = sum([t[1] for t in top_topics]) / args.num_topics
        logger.info(f'Average topic coherence: {avg_topic_coherence:.4f}.')
        for topic_idx, (topic_words, topic_score) in enumerate(top_topics):
            logger.info(f'Topic #{topic_idx} ({topic_score:.4f}): ' + " ".join((t[1] for t in topic_words[:5])))
        logger.info(
            f'Perplexity: {np.exp2(-model.log_perplexity(bow_corpus)):.4f}')
    else:
        pprint(model.show_topics(formatted=False))

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=model,
                                        texts=corpus,
                                        dictionary=dictionary,
                                        coherence=args.coherence,
                                        topn=args.topn)
    coherence_lda = coherence_model_lda.get_coherence()
    logger.info(f'Coherence : {coherence_lda:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LDA Model')
    parser.add_argument('--dataset_dir', required=True, help='dataset directory')
    parser.add_argument('--dump_dir', help='dump directory')
    parser.add_argument('--model', default='lda',
                        choices=['lda', 'multicore_lda', 'mallet_lda', 'gensim_lda', 'gensim_multicore_lda'],
                        help='which lda to use')
    parser.add_argument('--mallet-path', help='mallet path')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--num_topics', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--log_dir', type=str, help='tb directory')
    parser.add_argument('--callbacks', choices=['perplexity', 'coherence'], nargs='+', default=[])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--coherence', type=str, default='c_v', choices=['c_v', 'u_mass'], help='cohrence metrics')
    parser.add_argument('--topn', type=int, default=20)
    args = parser.parse_args()
    if not args.dump_dir:
        args.dump_dir = os.path.join(args.dataset_dir, 'lda_dump')
    if not os.path.exists(args.dump_dir):
        os.makedirs(args.dump_dir)
    if not args.log_dir:
        args.log_dir = os.path.join(args.dump_dir, 'logs')
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if args.model == 'mallet_lda' and not args.mallet_path:
        args.mallet_path = 'lib/mallet/mallet-2.0.8/bin/mallet'

    set_tee_logger(args.dump_dir)
    logger = logging.getLogger()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    print(args)
    main()
