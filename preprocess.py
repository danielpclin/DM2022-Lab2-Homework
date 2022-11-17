import json
import os
import pickle

import pandas as pd
import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizer


def preprocess():
    DATA_DIR = 'data'
    DATA_ID_PATH = f'{DATA_DIR}/data_identification.csv'
    EMOTION_PATH = f'{DATA_DIR}/emotion.csv'
    TWEETS_PATH = f'{DATA_DIR}/tweets_DM.json'
    SAMPLE_PATH = f'{DATA_DIR}/sampleSubmission.csv'
    TRAIN_PKL = f"data/train.pkl"
    TEST_PKL = f"data/test.pkl"

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


def adapt():
    VECTORIZER_PKL = f"data/vectorizer.pkl"
    train_pkl = f"data/train.pkl"
    print("Reading data")
    train_df = pd.read_pickle(train_pkl)
    print(f"{train_df.shape = }")

    print("Adapting text vectorization")
    max_features = 20000
    sequence_length = 128
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    vectorize_layer.adapt(train_df['text'])  # adapt dataset

    print("Saving text vectorization")
    pickle.dump({'config': vectorize_layer.get_config(), 'weights': vectorize_layer.get_weights()},
                open(VECTORIZER_PKL, "wb"))
    print(f"Max length: {np.max(np.count_nonzero(vectorize_layer(train_df['text']), axis=1))}")


def dbert_tokenize():
    pickle_input_path = './data/dbert_inputs.pkl'
    pickle_mask_path = './data/dbert_mask.pkl'
    pickle_label_path = './data/dbert_label.pkl'
    pickle_test_input_path = './data/dbert_test_inputs.pkl'
    pickle_test_mask_path = './data/dbert_test_mask.pkl'
    train_pkl = f"data/train.pkl"
    test_pkl = f"data/test.pkl"
    max_len = 256

    print("Reading data")
    train_df = pd.read_pickle(train_pkl)
    print(f"{train_df.shape = }")
    test_df = pd.read_pickle(test_pkl)
    print(f"{test_df.shape = }")

    train_x = train_df['text']
    train_y = train_df['emotion']
    dbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Encode train set
    print('Preparing the pickle file.....')
    input_ids = []
    attention_masks = []

    for sent in train_x:
        dbert_inputs = dbert_tokenizer.encode_plus(sent, add_special_tokens=True, max_length=max_len,
                                                   padding='max_length', return_attention_mask=True, truncation=True)
        input_ids.append(dbert_inputs['input_ids'])
        attention_masks.append(dbert_inputs['attention_mask'])

    input_ids = np.asarray(input_ids)
    attention_masks = np.array(attention_masks)
    labels = np.array(train_y)

    pickle.dump(input_ids, open(pickle_input_path, 'wb'))
    pickle.dump(attention_masks, open(pickle_mask_path, 'wb'))
    pickle.dump(labels, open(pickle_label_path, 'wb'))

    print('Pickle files saved as ', pickle_input_path, pickle_mask_path, pickle_label_path)

    # Encode test set
    test_x = test_df['text']

    test_inputs = []
    test_mask = []

    for sent in test_x:
        dbert_inputs = dbert_tokenizer.encode_plus(sent, add_special_tokens=True, max_length=max_len,
                                                   padding='max_length', return_attention_mask=True, truncation=True)
        test_inputs.append(dbert_inputs['input_ids'])
        test_mask.append(dbert_inputs['attention_mask'])

    test_inputs = np.asarray(test_inputs)
    test_mask = np.array(test_mask)

    pickle.dump(test_inputs, open(pickle_test_input_path, 'wb'))
    pickle.dump(test_mask, open(pickle_test_mask_path, 'wb'))



if __name__ == "__main__":
    preprocess()
    adapt()
    dbert_tokenize()
    pass
