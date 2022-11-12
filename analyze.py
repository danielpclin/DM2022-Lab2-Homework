import json
import os

import pandas as pd
from pandas_profiling import ProfileReport
import tqdm

DATA_DIR = 'data'

DATA_ID_PATH = f'{DATA_DIR}/data_identification.csv'
EMOTION_PATH = f'{DATA_DIR}/emotion.csv'
TWEETS_PATH = f'{DATA_DIR}/tweets_DM.json'
SAMPLE_PATH = f'{DATA_DIR}/sampleSubmission.csv'

print(f"Reading {EMOTION_PATH}")
print(f"Reading {DATA_ID_PATH}")
print(f"Reading {TWEETS_PATH}")
tweet = pd.read_json(TWEETS_PATH, lines=True)
tweet = pd.json_normalize(json.loads(tweet.to_json(orient='records')))
tweet = tweet.rename(columns={'_source.tweet.tweet_id': 'id', '_source.tweet.hashtags': 'hashtags', '_source.tweet.text': 'text'})
tweet = pd.read_csv(DATA_ID_PATH).join(tweet.set_index('id'), on='tweet_id')
tweet.drop(['_index', '_type'], inplace=True)
tweet_train = tweet[tweet['identification'] == 'train']
tweet_test = tweet[tweet['identification'] == 'test']
tweet_train = tweet_train.join(pd.read_csv(EMOTION_PATH).set_index('tweet_id'), on='tweet_id')

print(f"Reading {SAMPLE_PATH}")
sample = pd.read_csv(SAMPLE_PATH)

