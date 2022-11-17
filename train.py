import os
import pickle

from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertModel

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import xgboost as xgb
from wandb.integration.keras import WandbCallback

# Setup mixed precision

# tf.config.set_visible_devices([], 'GPU')
tf.keras.mixed_precision.set_global_policy('mixed_float16')


class MinimumEpochEarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None,
                 restore_best_weights=False, min_epoch=30):
        super(MinimumEpochEarlyStopping, self).__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights)
        self.min_epoch = min_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.min_epoch:
            super().on_epoch_end(epoch, logs)


class MinimumEpochReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self, monitor='val_loss', min_delta=0., patience=0, verbose=0, mode='auto', factor=0.1, cooldown=0,
                 min_lr=0., min_epoch=30):
        super(MinimumEpochReduceLROnPlateau, self).__init__(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr, )
        self.min_epoch = min_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.min_epoch:
            super().on_epoch_end(epoch, logs)


def plot(history, version_num):
    fig = plt.figure(figsize=(20, 15))

    # Plot training accuracy
    plt.subplot(2, 1, 1)
    training_accuracy_keys = [key for key in history.keys() if 'accuracy' in key and 'val' not in key]
    for key in training_accuracy_keys:
        plt.plot(history[key])
    plt.title('Model training accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(training_accuracy_keys)
    # plt.ylim(0.8, 1)
    plt.grid()

    # Plot training loss
    plt.subplot(2, 1, 2)
    training_loss_keys = [key for key in history.keys() if 'loss' in key and 'val' not in key]
    for key in training_loss_keys:
        plt.plot(history[key])
    plt.title('Model training loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(training_loss_keys)
    plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"save/{version_num}/train.png")
    plt.close(fig)

    fig = plt.figure(figsize=(20, 15))

    # Plot val accuracy
    plt.subplot(2, 1, 1)
    val_accuracy_keys = [key for key in history.keys() if 'accuracy' in key and 'val' in key]
    for key in val_accuracy_keys:
        plt.plot(history[key])
    plt.title('Model val accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(val_accuracy_keys)
    # plt.ylim(0.8, 1)
    plt.grid()

    # Plot val loss
    plt.subplot(2, 1, 2)
    val_loss_keys = [key for key in history.keys() if 'loss' in key and 'val' in key]
    for key in val_loss_keys:
        plt.plot(history[key])
    plt.title('Model val loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(val_loss_keys)
    plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"save/{version_num}/validation.png")
    plt.close(fig)


def train(version_num, batch_size=64):
    train_pkl = f"data/train.pkl"
    vectorizer_pkl = f"data/vectorizer.pkl"
    checkpoint_path = f'save/{version_num}/checkpoint.hdf5'
    log_dir = f'save/{version_num}'
    epochs = 100
    learning_rate = 0.001
    embedding_dim = 128
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    print("Reading data")
    train_df = pd.read_pickle(train_pkl)
    print(f"{train_df.shape = }")

    vectorizer_dict = pickle.load(open(vectorizer_pkl, "rb"))
    vectorize_layer = tf.keras.layers.TextVectorization.from_config(vectorizer_dict['config'])
    vectorize_layer.set_weights(vectorizer_dict['weights'])

    train_y = pd.get_dummies(train_df['emotion'])
    train_x, val_x, train_y, val_y = train_test_split(train_df['text'], train_y, test_size=0.1)
    train_x = vectorize_layer(train_x)
    val_x = vectorize_layer(val_x)

    input_shape = (vectorize_layer.get_config()['output_sequence_length'], )
    main_input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Embedding(vectorize_layer.get_config()['max_tokens'], embedding_dim)(main_input)
    x = tf.keras.layers.Conv1D(128, 3, padding="valid", activation="relu", strides=1)(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="valid", activation="relu", strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv1D(256, 3, padding="valid", activation="relu", strides=1)(x)
    x = tf.keras.layers.Conv1D(256, 3, padding="valid", activation="relu", strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv1D(512, 3, padding="valid", activation="relu", strides=1)(x)
    x = tf.keras.layers.Conv1D(512, 3, padding="valid", activation="relu", strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(8, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(main_input, out)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                                                    save_weights_only=False, mode='auto')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='auto')
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, cooldown=1,
                                                     mode='auto', min_lr=0.00001)
    callbacks_list = [checkpoint, tensor_board, early_stop, reduce_lr]

    try:
        train_history = model.fit(
            train_x, train_y,
            steps_per_epoch=np.ceil(train_x.shape[0] / batch_size),
            epochs=epochs,
            validation_data=(val_x, val_y),
            verbose=1,
            callbacks=callbacks_list
        )
    except KeyboardInterrupt:
        pass
    else:
        with open(f"save/{version_num}/result.pkl", "wb") as file:
            pickle.dump(train_history.history, file)
        with open(f"save/{version_num}/results.txt", "w") as file:
            loss_idx = np.nanargmin(train_history.history['val_loss'])
            file.write("Loss:\n")
            file.write(f"{train_history.history['val_loss'][loss_idx]}\n")
            file.write("Accuracy:\n")
            file.write(f"{train_history.history['val_accuracy'][loss_idx]}\n")
        plot(train_history.history, version_num)
    finally:
        pass
    tf.keras.backend.clear_session()


def train_xgboost():
    train_pkl = f"data/train.pkl"
    vectorizer_pkl = f"data/vectorizer.pkl"

    print("Reading data")
    train_df = pd.read_pickle(train_pkl)
    print(f"{train_df.shape = }")

    vectorizer_dict = pickle.load(open(vectorizer_pkl, "rb"))
    vectorize_layer = tf.keras.layers.TextVectorization.from_config(vectorizer_dict['config'])
    vectorize_layer.set_weights(vectorizer_dict['weights'])

    train_y = pd.get_dummies(train_df['emotion']).to_numpy().argmax(axis=1)
    train_x, val_x, train_y, val_y = train_test_split(train_df['text'], train_y, test_size=0.1)
    train_x = vectorize_layer(train_x)
    val_x = vectorize_layer(val_x)

    params = {
        "objective": "multi:softprob",
        "eta": 0.3,
        "max_depth": 6,
        "min_child_weight": 1,
        "verbosity": 1,
        "num_class": 8,
    }
    num_trees = 300
    gbm = xgb.train(params, xgb.DMatrix(train_x, train_y), num_trees)

    print("Make predictions on the test set")
    test_probs = gbm.predict(xgb.DMatrix(val_x))
    print(f"Accuracy: {np.mean(test_probs.argmax(axis=1) == val_y)}")


def train_bert(version_num, batch_size=32):
    train_pkl = f"data/train.pkl"
    pickle_inp_path = './data/dbert_inputs.pkl'
    pickle_mask_path = './data/dbert_mask.pkl'
    pickle_label_path = './data/dbert_label.pkl'
    checkpoint_path = f'save/bert_{version_num}/checkpoint.tf'
    log_dir = f'save/bert_{version_num}'
    epochs = 30
    max_len = 256

    print("Reading data")
    train_df = pd.read_pickle(train_pkl)
    print(f"{train_df.shape = }")

    train_y = train_df['emotion']
    num_classes = len(train_y.unique())

    print('Loading the saved pickle files..')
    input_ids = pickle.load(open(pickle_inp_path, 'rb'))
    attention_masks = pickle.load(open(pickle_mask_path, 'rb'))
    labels = pickle.load(open(pickle_label_path, 'rb'))
    labels = pd.get_dummies(labels)

    print(f'Input shape {input_ids.shape}')
    print(f'Attention mask shape {attention_masks.shape}')
    print(f'Input label shape {labels.shape}')
    train_inputs, val_inputs, train_label, val_label, train_mask, val_mask = train_test_split(input_ids, labels,
                                                                                              attention_masks,
                                                                                              test_size=0.1)

    print(f'Train inp shape {train_inputs.shape} Val input shape {val_inputs.shape}')
    print(f"Train label shape {train_label.shape} Val label shape {val_label.shape}")
    print(f"Train attention mask shape {train_mask.shape} Val attention mask shape {val_mask.shape}")

    # Setup model
    dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

    inputs = tf.keras.layers.Input(shape=(max_len,), dtype='int64')
    masks = tf.keras.layers.Input(shape=(max_len,), dtype='int64')
    dbert_layer = dbert_model(inputs, attention_mask=masks)[0][:, 0, :]
    dense = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(dbert_layer)
    dropout = tf.keras.layers.Dropout(0.5)(dense)
    pred = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))(dropout)
    model = tf.keras.Model(inputs=[inputs, masks], outputs=pred)

    loss = tf.keras.losses.CategoricalCrossentropy()
    metric = tf.keras.metrics.CategoricalAccuracy('accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                                                    save_weights_only=False, mode='auto')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, cooldown=1,
                                                     mode='auto', min_lr=0.00001)
    callbacks_list = [checkpoint, tensor_board, early_stop, reduce_lr]

    try:
        train_history = model.fit(
            [train_inputs, train_mask], train_label,
            steps_per_epoch=np.ceil(train_inputs.shape[0] / batch_size),
            epochs=epochs,
            validation_data=([val_inputs, val_mask], val_label),
            verbose=1,
            callbacks=callbacks_list
        )
    except KeyboardInterrupt:
        pass
    else:
        with open(f"save/{version_num}/result.pkl", "wb") as file:
            pickle.dump(train_history.history, file)
        with open(f"save/{version_num}/results.txt", "w") as file:
            loss_idx = np.nanargmin(train_history.history['val_loss'])
            file.write("Loss:\n")
            file.write(f"{train_history.history['val_loss'][loss_idx]}\n")
            file.write("Accuracy:\n")
            file.write(f"{train_history.history['val_accuracy'][loss_idx]}\n")
        plot(train_history.history, version_num)
    finally:
        pass
    tf.keras.backend.clear_session()


def main():
    try:
        with open('save/run.txt') as file:
            version_num = int(file.readline())
    except (FileNotFoundError, ValueError):
        version_num = 0
    # train(version_num=version_num, batch_size=128)
    # train_xgboost()
    train_bert(version_num=version_num, batch_size=32)
    with open('save/run.txt', 'w') as file:
        file.write(f"{version_num+1}\n")


if __name__ == "__main__":
    for _ in range(2):
        main()
