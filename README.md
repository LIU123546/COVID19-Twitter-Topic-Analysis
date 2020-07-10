# COVID-19 Twitter Topics Analysis

## Dependencies

- nltk == 3.5
- gensim == 3.8.3
- mallet == 2.0.8

## Install

```bash
# create virtual env
conda create -n covid python=3
# install dependency
pip install -r requirements.txt
# download COVID confirmed cases data
git clone https://github.com/CSSEGISandData/COVID-19 lib/COVID-19
# downliad COVID-19 tweet IDs
git clone https://github.com/echen102/COVID-19-TweetIDs lib/COVID-19-TweetIDs
```

## Data Acquisition and Processing

```bash
# get tweet full text from tweet IDs
python lib/COVID-19-TweetIDs/hydrate.py
# filter tweets by country and language
python filter_tweets.py --input_dirs lib/COVID-19-TweetIDs/2020-{01,02,03,04} --output_dir data/COVID-19-Tweets-geo
# preprocessing tweets
python preprocess.py --input_dir data/COVID-19-Tweets-geo --output_dir data/COVID-19-Tweets-geo
# extract candidates
python extract_candidates.py --dataset_dir data/COVID-19-Tweets-geo
```

## Train LDA

```bash
python trainlda.py --datasetdir data/COVID-19-Tweets-geo --dump_dir dump/sample_mallet_lda --model mallet_lda --iterations 2000 --num_topics 20
```

## Analysis

[Data Analysis Notebook](./inspect_data.ipynb)

[Cohrence Analysis Notebook](./analyze_coherence.ipynb)
<!-- [Cohrence Analysis Notebook](https://nbviewer.jupyter.org/gist/Shuailong/414943ef4dbbe049b035d85a10e3b602) -->

[Predictions Analysis Notebook](./analyze_prediction.ipynb)
<!-- [Predictions Analysis Notebook](https://nbviewer.jupyter.org/gist/Shuailong/41bd8fae0a80758ef7cd814df91d7cd4) -->

[PyLDAvis Notebook](./pyldavis.ipynb)
<!-- [PyLDAvis Notebook](https://nbviewer.jupyter.org/gist/Shuailong/32942e5703817d4cc130f74afbd0be33) -->
