import os
import pickle

import pandas as pd
import tensorflow as tf
import numpy as np
from transformers import TFDistilBertModel


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def predict(version_num=1):
    save_path = f'save/{version_num}'
    checkpoint_path = f'{save_path}/checkpoint.hdf5'
    test_pkl = f"data/test.pkl"
    vectorizer_pkl = f"data/vectorizer.pkl"

    with open('data/emotion_encodings.pkl', 'rb') as file:
        encodings = pickle.load(file)

    test_df = pd.read_pickle(test_pkl)
    print(f"{test_df.shape = }")

    vectorizer_dict = pickle.load(open(vectorizer_pkl, "rb"))
    vectorize_layer = tf.keras.layers.TextVectorization.from_config(vectorizer_dict['config'])
    vectorize_layer.set_weights(vectorizer_dict['weights'])

    test_x = vectorize_layer(test_df['text'])

    model = tf.keras.models.load_model(checkpoint_path)
    model.summary()

    pred = model.predict(test_x)

    pred = np.argmax(pred, axis=1)
    test_df['emotion'] = pred
    test_df['emotion'] = test_df['emotion'].apply(lambda x: encodings[x])
    test_df.rename(columns={'tweet_id': 'id'}).to_csv(f'{save_path}/prediction.csv', index=False, columns=['id', 'emotion'])


def predict_bert(version_num=1):
    save_path = f'save/bert_{version_num}'
    checkpoint_path = f'{save_path}/checkpoint.tf'
    pickle_input_path = './data/dbert_test_inputs.pkl'
    pickle_mask_path = './data/dbert_test_mask.pkl'

    with open('data/emotion_encodings.pkl', 'rb') as file:
        encodings = pickle.load(file)

    test_pkl = f"data/test.pkl"
    print("Reading data")
    test_df = pd.read_pickle(test_pkl)
    print(f"{test_df.shape = }")

    model = tf.keras.models.load_model(checkpoint_path)
    model.summary()

    print('Loading the saved pickle files..')
    test_inputs = pickle.load(open(pickle_input_path, 'rb'))
    test_mask = pickle.load(open(pickle_mask_path, 'rb'))

    print(f'Test input shape {test_inputs.shape}')
    print(f"Test attention mask shape {test_mask.shape}")

    pred = model.predict([test_inputs, test_mask])

    pred = np.argmax(pred, axis=1)
    test_df['emotion'] = pred
    test_df['emotion'] = test_df['emotion'].apply(lambda x: encodings[x])
    test_df.rename(columns={'tweet_id': 'id'}).to_csv(f'{save_path}/prediction.csv', index=False, columns=['id', 'emotion'])


def main():
    # predict(20)
    predict_bert()


if __name__ == "__main__":
    main()
