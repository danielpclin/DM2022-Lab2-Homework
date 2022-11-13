import itertools
import os
import pickle

import pandas as pd
import tensorflow as tf
import numpy as np
from functools import reduce

# Setup mixed precision
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.keras.mixed_precision.set_global_policy('mixed_float16')

predicted_dict = {}


def predict(versions=(1,), batch_size=64, method="occur_max", evaluate=False):
    train_pkl = f"data/train.pkl"
    test_pkl = f"data/test.pkl"
    vectorizer_pkl = f"data/vectorizer.pkl"

    vectorizer_dict = pickle.load(open(vectorizer_pkl, "rb"))
    vectorize_layer = tf.keras.layers.TextVectorization.from_config(vectorizer_dict['config'])
    vectorize_layer.set_weights(vectorizer_dict['weights'])

    with open('data/emotion_encodings.pkl', 'rb') as file:
        encodings = pickle.load(file)

    if evaluate:
        df = pd.read_pickle(train_pkl)
    else:
        df = pd.read_pickle(test_pkl)

    predictions = []
    for version in versions:
        save_path = f'save/{version}'
        checkpoint_path = f'{save_path}/checkpoint.hdf5'
        model = tf.keras.models.load_model(checkpoint_path)

        inputs = tf.keras.Input(shape=(1,), dtype="string")
        x = vectorize_layer(inputs)
        outputs = model(x)

        end_to_end_model = tf.keras.Model(inputs, outputs)
        end_to_end_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        _prediction = end_to_end_model.predict(df['text'], batch_size=batch_size)
        tf.keras.backend.clear_session()
        predictions.append(_prediction)

    if len(versions) == 1:
        prediction = np.argmax(predictions[0], axis=1)
        df['emotion'] = prediction
        df['emotion'] = df['emotion'].apply(lambda x: encodings[x])
    else:
        prediction_sum_argmax = np.argmax(reduce(np.add, predictions), axis=1)
        prediction_argmax_concat = np.concatenate(np.expand_dims(np.argmax(predictions, axis=2), axis=2), axis=1)
        prediction_concat_argmax = np.argmax(np.concatenate(predictions, axis=1), axis=1)
        if method == "occur_max":
            pass
            # for letter_index, letter in enumerate(prediction_argmax_concat):
            #     for label_index, label in enumerate(letter):
            #         (values, counts) = np.unique(label, return_counts=True)
            #         label = values[counts == counts.max()]
            #         if len(label) > 1:
            #             result[label_index] = result[label_index] + int_to_char[
            #                 prediction_concat_argmax[letter_index][label_index] % len(alphabet)]
            #         else:
            #             result[label_index] = result[label_index] + int_to_char[label[0] % len(alphabet)]
        elif method == "occur_sum_max":
            pass
            # for letter_index, letter in enumerate(prediction_argmax_concat):
            #     for label_index, label in enumerate(letter):
            #         (values, counts) = np.unique(label, return_counts=True)
            #         label = values[counts == counts.max()]
            #         if len(label) > 1:
            #             result[label_index] = result[label_index] + int_to_char[
            #                 prediction_sum_argmax[letter_index][label_index] % len(alphabet)]
            #         else:
            #             result[label_index] = result[label_index] + int_to_char[label[0] % len(alphabet)]
        elif method == "max":
            pass
            # for letter in prediction_concat_argmax:
            #     for index, label in enumerate(letter):
            #         result[index] = result[index] + int_to_char[label % len(alphabet)]
        elif method == "sum_max":
            pass
            # for letter in prediction_sum_argmax:
            #     for index, label in enumerate(letter):
            #         result[index] = result[index] + int_to_char[label % len(alphabet)]

    df.rename(columns={'tweet_id': 'id'}).to_csv(f'save/{"_".join(versions)}/prediction.csv', index=False, columns=['id', 'emotion'])
    if len(versions) == 1:
        df.rename(columns={'tweet_id': 'id'}).to_csv(f'save/{versions[0]}/prediction.csv', index=False, columns=['id', 'emotion'])
    else:
        df.rename(columns={'tweet_id': 'id'}).to_csv(f'save/{"_".join(map(str, versions))}_{method}_prediction.csv', index=False, columns=['id', 'emotion'])



    # for version in versions:
    #     filename = f"predicted_result/{'eval_' if evaluate else ''}{version}.pickle"
    #     if version in predicted_dict:
    #         _prediction = predicted_dict[version]
    #     elif os.path.isfile(filename):
    #         with open(filename, 'rb') as f:
    #             _prediction = pickle.load(f)
    #             predicted_dict[version] = _prediction
    #     else:
    #         checkpoint_path = f'checkpoints/{version}.hdf5'
    #         image_data_generator = ImageDataGenerator(rescale=1. / 255)
    #         predict_generator = image_data_generator.flow_from_dataframe(dataframe=df, directory=predict_dataset_dir,
    #                                                                      x_col=x_col, class_mode=None, shuffle=False,
    #                                                                      target_size=(img_height, img_width),
    #                                                                      batch_size=batch_size)
    #         model = models.load_model(checkpoint_path)
    #         _prediction = model.predict(
    #             predict_generator,
    #             steps=np.ceil(predict_generator.n / predict_generator.batch_size),
    #             verbose=1,
    #         )
    #         K.clear_session()
    #         with open(filename, 'wb') as f:
    #             pickle.dump(_prediction, f)
    #         predicted_dict[version] = _prediction
    #     predictions.append(_prediction)

def main():
    predict(versions=(23, 24), batch_size=64, method="sum_max", evaluate=False)


if __name__ == "__main__":
    main()
