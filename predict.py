import os
import pickle

import pandas as pd
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def predict(version_num=1):
    save_path = f'save/{version_num}'
    checkpoint_path = f'{save_path}/checkpoint.hdf5'
    train_pkl = f"data/train.pkl"
    test_pkl = f"data/test.pkl"
    vectorizer_pkl = f"data/vectorizer.pkl"

    train_df = pd.read_pickle(train_pkl)
    encodings = pd.get_dummies(train_df['emotion']).columns

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


def main():
    predict(2)


if __name__ == "__main__":
    main()
