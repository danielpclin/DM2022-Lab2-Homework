import pickle

import pandas as pd
import tensorflow as tf
import numpy as np
from functools import reduce

from transformers import TFDistilBertModel, TFRobertaModel

# Setup mixed precision
# tf.config.set_visible_devices([], 'GPU')
tf.keras.mixed_precision.set_global_policy('mixed_float16')


def predict_dbert(versions=(1,), batch_size=32):
    print('Preparing the pickle file.....')
    pickle_input_path = './data/dbert_test_inputs.pkl'
    pickle_mask_path = './data/dbert_test_mask.pkl'

    print('Loading the saved pickle files..')
    test_inputs = pickle.load(open(pickle_input_path, 'rb'))
    test_mask = pickle.load(open(pickle_mask_path, 'rb'))

    print(f'Test input shape {test_inputs.shape}')
    print(f"Test attention mask shape {test_mask.shape}")

    predictions = []
    for version in versions:
        print(f"Predicting bert {version}...")
        save_path = f'save/{version}_bert'
        try:
            with open(f"{save_path}/predict.pkl", "rb") as file:
                print(f"file found, skipping...")
                _prediction = pickle.load(file)
                predictions.append(_prediction)
        except FileNotFoundError:
            pass
        else:
            continue
        print(f"file not found, predicting with model...")
        checkpoint_path = f'{save_path}/checkpoint.hdf5'
        model = tf.keras.models.load_model(checkpoint_path, custom_objects={"TFDistilBertModel": TFDistilBertModel})

        _prediction = model.predict([test_inputs, test_mask], batch_size=batch_size)
        tf.keras.backend.clear_session()
        with open(f"{save_path}/predict.pkl", "wb") as file:
            pickle.dump(_prediction, file)
        predictions.append(_prediction)
    return predictions

def predict_roberta(versions=(1,), batch_size=32):
    print('Preparing the pickle file.....')
    name = 'roberta'
    pickle_input_path = f'./data/{name}_test_inputs.pkl'
    pickle_mask_path = f'./data/{name}_test_mask.pkl'

    print('Loading the saved pickle files..')
    test_inputs = pickle.load(open(pickle_input_path, 'rb'))
    test_mask = pickle.load(open(pickle_mask_path, 'rb'))

    print(f'Test input shape {test_inputs.shape}')
    print(f"Test attention mask shape {test_mask.shape}")

    predictions = []
    for version in versions:
        print(f"Predicting roberta {version}...")
        save_path = f'save/{version}_{name}'
        try:
            with open(f"{save_path}/predict.pkl", "rb") as file:
                print(f"file found, skipping...")
                _prediction = pickle.load(file)
                predictions.append(_prediction)
        except FileNotFoundError:
            pass
        else:
            continue
        print(f"file not found, predicting with model...")
        checkpoint_path = f'{save_path}/checkpoint.hdf5'
        model = tf.keras.models.load_model(checkpoint_path, custom_objects={"TFRobertaModel": TFRobertaModel})

        _prediction = model.predict([test_inputs, test_mask], batch_size=batch_size, verbose=1)
        tf.keras.backend.clear_session()
        with open(f"{save_path}/predict.pkl", "wb") as file:
            pickle.dump(_prediction, file)
        predictions.append(_prediction)
    return predictions


def predict_cnn(df, versions=(1,), batch_size=64):
    vectorizer_pkl = f"data/vectorizer.pkl"
    vectorizer_dict = pickle.load(open(vectorizer_pkl, "rb"))
    vectorize_layer = tf.keras.layers.TextVectorization.from_config(vectorizer_dict['config'])
    vectorize_layer.set_weights(vectorizer_dict['weights'])

    predictions = []
    for version in versions:
        print(f"Predicting {version}...")
        save_path = f'save/{version}'
        try:
            with open(f"{save_path}/predict.pkl", "rb") as file:
                print(f"file found, skipping...")
                _prediction = pickle.load(file)
                predictions.append(_prediction)
        except FileNotFoundError:
            pass
        else:
            continue
        print(f"file not found, predicting with model...")
        checkpoint_path = f'{save_path}/checkpoint.hdf5'
        model = tf.keras.models.load_model(checkpoint_path)

        inputs = tf.keras.Input(shape=(1,), dtype="string")
        x = vectorize_layer(inputs)
        outputs = model(x)

        end_to_end_model = tf.keras.Model(inputs, outputs)
        end_to_end_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        _prediction = end_to_end_model.predict(df['text'], batch_size=batch_size)
        tf.keras.backend.clear_session()
        with open(f"{save_path}/predict.pkl", "wb") as file:
            pickle.dump(_prediction, file)
        predictions.append(_prediction)
    return predictions


def predict(versions=None, bert_versions=None, roberta_versions=None, method="occur_max"):
    test_pkl = f"data/test.pkl"

    with open('data/emotion_encodings.pkl', 'rb') as file:
        encodings = pickle.load(file)

    df = pd.read_pickle(test_pkl)

    predictions = []
    if versions is not None:
        predictions += predict_cnn(df, versions, 128)
    if bert_versions is not None:
        predictions += predict_dbert(bert_versions, 32)
    if roberta_versions is not None:
        predictions += predict_roberta(roberta_versions, 16)

    print("Ensembling...")
    if len(predictions) == 1:
        prediction = np.argmax(predictions[0], axis=1)
        df['emotion'] = prediction
        df['emotion'] = df['emotion'].apply(lambda x: encodings[x])
    else:
        prediction_sum_argmax = np.argmax(reduce(np.add, predictions), axis=1)
        prediction_argmax_concat = np.concatenate(np.expand_dims(np.argmax(predictions, axis=2), axis=2), axis=1)
        prediction_concat_argmax = np.argmax(np.concatenate(predictions, axis=1), axis=1)
        result = pd.Series("", index=np.arange(prediction_argmax_concat.shape[0]))
        if method == "occur_max":
            for index, label in enumerate(prediction_argmax_concat):
                (values, counts) = np.unique(label, return_counts=True)
                label = values[counts == counts.max()]
                if len(label) > 1:
                    result[index] = encodings[prediction_concat_argmax[index] % len(encodings)]
                else:
                    result[index] = encodings[label[0] % len(encodings)]
        elif method == "occur_sum_max":
            for index, label in enumerate(prediction_argmax_concat):
                (values, counts) = np.unique(label, return_counts=True)
                label = values[counts == counts.max()]
                if len(label) > 1:
                    result[index] = encodings[prediction_sum_argmax[index] % len(encodings)]
                else:
                    result[index] = encodings[label[0] % len(encodings)]
        elif method == "max":
            for index, label in enumerate(prediction_concat_argmax):
                result[index] = encodings[label % len(encodings)]
        elif method == "sum_max":
            for index, label in enumerate(prediction_sum_argmax):
                result[index] = encodings[label % len(encodings)]
        df.reset_index(inplace=True)
        df['emotion'] = result

    if len(predictions) == 1:
        if versions is not None:
            df.rename(columns={'tweet_id': 'id'}).to_csv(f'save/{f"{versions[0]}"}/prediction.csv', index=False, columns=['id', 'emotion'])
        if bert_versions is not None:
            df.rename(columns={'tweet_id': 'id'}).to_csv(f'save/{f"{bert_versions[0]}_bert"}/prediction.csv', index=False, columns=['id', 'emotion'])
        if roberta_versions is not None:
            df.rename(columns={'tweet_id': 'id'}).to_csv(f'save/{f"{roberta_versions[0]}_roberta"}/prediction.csv', index=False, columns=['id', 'emotion'])
    else:
        all_versions = tuple()
        if versions is not None:
            all_versions += versions
        if bert_versions is not None:
            all_versions += bert_versions
        if roberta_versions is not None:
            all_versions += roberta_versions
        df.rename(columns={'tweet_id': 'id'}).to_csv(f'save/{"_".join(map(str, all_versions))}_{method}_prediction.csv',
                                                     index=False, columns=['id', 'emotion'])


def main():
    # predict(versions=(57,), bert_versions=(44, 45, 58), roberta_versions=(60,), method="occur_max")
    predict(bert_versions=(44, 45, 58), roberta_versions=(60,), method="occur_max")


if __name__ == "__main__":
    main()
