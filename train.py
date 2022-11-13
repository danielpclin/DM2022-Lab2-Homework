import os
import pickle

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging

# Setup mixed precision
from wandb.integration.keras import WandbCallback

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
    plt.ylim(0.8, 1)
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
    plt.ylim(0.8, 1)
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

    log = logging.getLogger('tensorflow')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(f'save/{version_num}/run.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    print("Reading data")
    train_df = pd.read_pickle(train_pkl)
    print(f"{train_df.shape = }")

    vectorizer_dict = pickle.load(open(vectorizer_pkl, "rb"))
    vectorize_layer = tf.keras.layers.TextVectorization.from_config(vectorizer_dict['config'])
    vectorize_layer.set_weights(vectorizer_dict['weights'])

    train_x = vectorize_layer(train_df['text'])
    train_y = pd.get_dummies(train_df['emotion'])

    input_shape = (vectorize_layer.get_config()['output_sequence_length'], )
    main_input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Embedding(vectorize_layer.get_config()['max_tokens'], embedding_dim)(main_input)
    x = tf.keras.layers.Conv1D(128, 3, padding="valid", activation="relu", strides=1)(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="valid", activation="relu", strides=1)(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="valid", activation="relu", strides=1)(x)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv1D(256, 3, padding="valid", activation="relu", strides=1)(x)
    x = tf.keras.layers.Conv1D(256, 3, padding="valid", activation="relu", strides=1)(x)
    x = tf.keras.layers.Conv1D(256, 3, padding="valid", activation="relu", strides=1)(x)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(8, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(main_input, out)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                                                    save_weights_only=False, mode='auto')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    # early_stop = MinimumEpochEarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', min_epoch=1)
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, cooldown=1,
                                                     mode='auto', min_lr=0.00001)
    # reduce_lr = MinimumEpochReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, cooldown=1, mode='auto',
    #                                           min_lr=0.00001, min_epoch=1)
    run = wandb.init(project="twitter_sentiment", entity="danielpclin", reinit=True, config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "version": version_num,
        "optimizer": optimizer._name
    })
    wandb_callback = WandbCallback()
    callbacks_list = [tensor_board, early_stop, checkpoint, reduce_lr, wandb_callback]

    try:
        train_history = model.fit(
            train_x, train_y,
            steps_per_epoch=np.ceil(train_x.shape[0] // batch_size),
            epochs=epochs,
            validation_split=0.2,
            verbose=1,
            callbacks=callbacks_list
        )
    except KeyboardInterrupt:
        pass
    else:
        with open(f"save/{version_num}/result.pickle", "wb") as file:
            pickle.dump(train_history.history, file)
        with open(f"save/{version_num}/results.txt", "w") as file:
            loss_idx = np.nanargmin(train_history.history['val_loss'])
            file.write("Loss:\n")
            file.write(f"{train_history.history['val_loss'][loss_idx]}\n")
            file.write("Accuracy:\n")
            file.write(f"{train_history.history['val_accuracy'][loss_idx]}\n")
        plot(train_history.history, version_num)
    finally:
        run.finish()
    tf.keras.backend.clear_session()

    # x = main_input
    # # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    # # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    # # x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), activation='relu')(x)
    # # x = Conv2D_BN_Activation(filters=64, kernel_size=(7, 7))(x)
    # # x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    # # x = Residual_Block(filters=64, kernel_size=(3, 3))(x)
    # # x = Residual_Block(filters=64, kernel_size=(3, 3))(x)
    # # x = Residual_Block(filters=64, kernel_size=(3, 3))(x)
    # # x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    # # x = tf.keras.layers.Dropout(0.2)(x)
    # # x = Residual_Block(filters=128, kernel_size=(3, 3), with_conv_shortcut=True)(x)
    # # x = Residual_Block(filters=128, kernel_size=(3, 3))(x)
    # # x = Residual_Block(filters=128, kernel_size=(3, 3))(x)
    # # x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    # # x = tf.keras.layers.Dropout(0.2)(x)
    # # x = Residual_Block(filters=256, kernel_size=(3, 3), with_conv_shortcut=True)(x)
    # # x = Residual_Block(filters=256, kernel_size=(3, 3))(x)
    # # x = Residual_Block(filters=256, kernel_size=(3, 3))(x)
    # # x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    # # x = tf.keras.layers.Dropout(0.3)(x)
    # # x = Conv2D_BN_Activation(filters=256, kernel_size=(3, 3))(x)
    # # x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), padding='same')(x)
    # # x = tf.keras.layers.Dropout(0.3)(x)
    # # x = tf.keras.layers.Flatten()(x)
    # out = [tf.keras.layers.Dense(len(alphabet), name=f'label{i}', activation='softmax')(x) for i in range(1, 13)]
    # model = tf.keras.Model(main_input, out)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


def main():
    train(version_num=3, batch_size=128)
    # for i in range(4, 10):
    #     train(version_num=i, batch_size=100)


if __name__ == "__main__":
    main()
