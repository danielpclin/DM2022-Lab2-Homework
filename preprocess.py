import json
import os
import pickle

import pandas as pd
import tensorflow as tf
import numpy as np


DATA_DIR = 'data'
DATA_ID_PATH = f'{DATA_DIR}/data_identification.csv'
EMOTION_PATH = f'{DATA_DIR}/emotion.csv'
TWEETS_PATH = f'{DATA_DIR}/tweets_DM.json'
SAMPLE_PATH = f'{DATA_DIR}/sampleSubmission.csv'
TRAIN_PKL = f"data/train.pkl"
TEST_PKL = f"data/test.pkl"
VECTORIZER_PKL = f"data/vectorizer.pkl"

print(f"Reading data...")
sample = pd.read_csv(SAMPLE_PATH)
tweet = pd.read_json(TWEETS_PATH, lines=True)
tweet = pd.json_normalize(json.loads(tweet.to_json(orient='records')))
tweet = tweet.rename(columns={
    '_source.tweet.tweet_id': 'id',
    '_source.tweet.hashtags': 'hashtags',
    '_source.tweet.text': 'text',
    '_score': 'score',
    '_crawldate': 'crawl_date'
})
tweet = pd.read_csv(DATA_ID_PATH).join(tweet.set_index('id'), on='tweet_id', validate='1:1')
tweet.drop(columns=['_index', '_type', 'score', 'crawl_date'], inplace=True)
tweet_train = tweet[tweet['identification'] == 'train']
tweet_train = tweet_train.drop(columns=['identification'])
tweet_test = tweet[tweet['identification'] == 'test']
tweet_test = tweet_test.drop(columns=['identification'])
tweet_train = tweet_train.join(pd.read_csv(EMOTION_PATH).set_index('tweet_id'), on='tweet_id', validate='1:1')
print("Saving dataframes")
tweet_train.to_pickle(TRAIN_PKL)
tweet_test.to_pickle(TEST_PKL)

# print("Adapting text vectorization")
# max_features = 20000
# sequence_length = 128
# vectorize_layer = tf.keras.layers.TextVectorization(
#     max_tokens=max_features,
#     output_mode="int",
#     output_sequence_length=sequence_length,
# )
# vectorize_layer.adapt(tweet_train['text'])  # adapt dataset
#
# print("Saving text vectorization")
# pickle.dump({'config': vectorize_layer.get_config(), 'weights': vectorize_layer.get_weights()},
#             open(VECTORIZER_PKL, "wb"))
# print(f"Max length: {np.max(np.count_nonzero(vectorize_layer(tweet_train['text']), axis=1))}")

encodings = pd.get_dummies(tweet_train['emotion']).columns
with open('data/emotion_encodings.pkl', 'wb') as file:
    pickle.dump(encodings, file)


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(f"train shape: {tweet_train.shape}")
    print(tweet_train.sample(10))
    print(tweet_train.memory_usage())
    print(f"test shape: {tweet_test.shape}")
    print(tweet_test.sample(10))
    print(tweet_test.memory_usage())
    print(f"test shape: {sample.shape}")
    print(sample.sample(10))
