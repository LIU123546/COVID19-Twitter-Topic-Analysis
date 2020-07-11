# COVID-19 Twitter Topics Analysis

## Dependencies

- twarc == 1.10.1 (download tweet from tweet id)
- nltk == 3.5 (preprocessing)
- gensim == 3.8.3 (LDA)
- mallet == 2.0.8 (LDA)

## Install

```bash
# clone this repository and cd to project root dir
git clone git@github.com:Shuailong/COVID19-Twitter-Topic-Analysis.git
cd COVID19-Twitter-Topic-Analysis
# create virtual env
conda create -n covid python=3
# install dependency
pip install -r requirements.txt
# download COVID confirmed cases data
git clone https://github.com/CSSEGISandData/COVID-19 lib/COVID-19
# downliad COVID-19 tweet IDs
git clone https://github.com/echen102/COVID-19-TweetIDs lib/COVID-19-TweetIDs
# download mallet lda
wget -O lib/mallet-2.0.8.zip http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
unzip lib/mallet-2.0.8.zip -d lib/mallet
```

## Data Acquisition and Processing

```bash
# get tweet full text from tweet IDs
cd lib/COVID-19-TweetIDs
# edit hydrate.py to select months
python hydrate.py
cd ../..
# filter tweets by country and language
mkdir -p data/COVID-19-Tweets-geo/2020-{01,02,03,04}
python filter_tweets.py --input_dirs lib/COVID-19-TweetIDs/2020-{01,02,03,04} --output_dir data/COVID-19-Tweets-geo
# preprocessing tweets
python preprocess.py --input_dir data/COVID-19-Tweets-geo --output_dir data/COVID-19-Tweets-geo --compress
# extract candidates
python extract_candidates.py --dataset_dir data/COVID-19-Tweets-geo
```

## Train LDA

```bash
python train_lda.py --dataset_dir data/COVID-19-Tweets-geo --dump_dir dump/sample_mallet_lda --model mallet_lda --iterations 2000 --num_topics 20
```

## Analysis

[Data Analysis Notebook](./inspect_data.ipynb)

[Cohrence Analysis Notebook](./analyze_coherence.ipynb)
<!-- [Cohrence Analysis Notebook](https://nbviewer.jupyter.org/gist/Shuailong/414943ef4dbbe049b035d85a10e3b602) -->

[Predictions Analysis Notebook](./analyze_prediction.ipynb)
<!-- [Predictions Analysis Notebook](https://nbviewer.jupyter.org/gist/Shuailong/41bd8fae0a80758ef7cd814df91d7cd4) -->

[PyLDAvis Notebook](./pyldavis.ipynb)
<!-- [PyLDAvis Notebook](https://nbviewer.jupyter.org/gist/Shuailong/32942e5703817d4cc130f74afbd0be33) -->
