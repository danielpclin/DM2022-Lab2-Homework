import os
import pickle

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Setup mixed precision
from wandb.integration.keras import WandbCallback

# tf.config.set_visible_devices([], 'GPU')
tf.keras.mixed_precision.set_global_policy('mixed_float16')


# Define residual blocks
def Conv2D_BN_Activation(filters, kernel_size, padding='same', strides=(1, 1), name=None):
    def block(input_x):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, name=conv_name)(input_x)
        x = tf.keras.layers.BatchNormalization(name=bn_name)(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    return block


def Conv2D_Activation_BN(filters, kernel_size, padding='same', strides=(1, 1), name=None):
    def block(input_x):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, name=conv_name)(input_x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name)(x)
        return x

    return block


# Define Residual Block
def Residual_Block(filters, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    def block(input_x):
        x = Conv2D_BN_Activation(filters=filters, kernel_size=(3, 3), padding='same')(input_x)
        x = Conv2D_BN_Activation(filters=filters, kernel_size=(3, 3), padding='same')(x)
        # need convolution on shortcut for add different channel
        if with_conv_shortcut:
            shortcut = Conv2D_BN_Activation(filters=filters, strides=strides, kernel_size=kernel_size)(input_x)
            x = tf.keras.layers.Add()([x, shortcut])
        else:
            x = tf.keras.layers.Add()([x, input_x])
        return x

    return block


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
    plt.savefig(f"results/train_{version_num}.png")
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
    plt.savefig(f"results/val_{version_num}.png")
    plt.close(fig)


def train(version_num, batch_size=64):
    train_pkl = f"data/train.h5"
    test_pkl = f"data/test.h5"
    vectorizer_pkl = f"data/vectorizer.pkl"
    checkpoint_path = f'save/{version_num}.hdf5'
    log_dir = f'save/{version_num}.log'
    epochs = 100
    learning_rate = 0.001
    embedding_dim = 256

    print("Reading data")
    # noinspection PyTypeChecker
    train_df = pd.read_pickle(train_pkl)
    # noinspection PyTypeChecker
    test_df = pd.read_pickle(test_pkl)
    print(f"{train_df.shape = }")
    print(f"{test_df.shape = }")

    vectorizer_dict = pickle.load(open(vectorizer_pkl, "rb"))
    vectorize_layer = tf.keras.layers.TextVectorization.from_config(vectorizer_dict['config'])
    vectorize_layer.set_weights(vectorizer_dict['weights'])

    vectorized_train_text = vectorize_layer(train_df['text'])
    # vectorized_test_text = vectorize_layer(test_df['text'])


    # optimizer = tf.keras.optimizers.Adam(learning_rate)
    # run = wandb.init(project="china_steel_ocr", entity="danielpclin", reinit=True, config={
    #     "learning_rate": learning_rate,
    #     "epochs": epochs,
    #     "batch_size": batch_size,
    #     "version": version_num,
    #     "optimizer": optimizer._name
    # })
    # img_width = 1232
    # img_height = 1028
    # alphabet = list('ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789 ')
    # char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    # int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # df = pd.read_csv(f'{training_dataset_csv}', delimiter=',')
    # df['filename'] = df['filename'] + '.jpg'
    # df['label'] = df['label'].apply(lambda el: list(el.ljust(12)))
    # df[[f'label{i}' for i in range(1, 13)]] = pd.DataFrame(df['label'].to_list(), index=df.index)
    # for i in range(1, 13):
    #     df[f'label{i}'] = df[f'label{i}'].apply(lambda el: tf.keras.utils.to_categorical(char_to_int[el], len(alphabet)))
    # data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.1)
    # train_generator = data_gen.flow_from_dataframe(dataframe=df, directory=training_dataset_dir, subset='training',
    #                                                x_col="filename", y_col=[f'label{i}' for i in range(1, 13)],
    #                                                class_mode="multi_output",
    #                                                target_size=(img_height, img_width), batch_size=batch_size)
    # validation_generator = data_gen.flow_from_dataframe(dataframe=df, directory=training_dataset_dir,
    #                                                     subset='validation',
    #                                                     x_col="filename", y_col=[f'label{i}' for i in range(1, 13)],
    #                                                     class_mode="multi_output",
    #                                                     target_size=(img_height, img_width), batch_size=batch_size)
    # input_shape = (img_height, img_width, 3)
    # main_input = tf.keras.layers.Input(shape=input_shape)
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
    # x = tf.keras.layers.Embedding(max_features + 1, embedding_dim)(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # out = [tf.keras.layers.Dense(len(alphabet), name=f'label{i}', activation='softmax')(x) for i in range(1, 13)]
    # model = tf.keras.Model(main_input, out)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
    #                              save_weights_only=False, mode='auto')
    #
    # early_stop = MinimumEpochEarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', min_epoch=20)
    # tensor_board = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # reduce_lr = MinimumEpochReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, cooldown=1, mode='auto',
    #                                           min_lr=0.00001, min_epoch=15)
    # wandb_callback = WandbCallback()
    # callbacks_list = [tensor_board, early_stop, checkpoint, reduce_lr, wandb_callback]
    #
    # model.summary()
    # train_history = model.fit(
    #     train_generator,
    #     steps_per_epoch=np.ceil(train_generator.n // train_generator.batch_size),
    #     epochs=epochs,
    #     validation_data=validation_generator,
    #     validation_steps=np.ceil(validation_generator.n // validation_generator.batch_size),
    #     verbose=1,
    #     callbacks=callbacks_list
    # )
    # with open(f"results/{version_num}.pickle", "wb") as file:
    #     pickle.dump(train_history.history, file)
    # with open(f"results/{version_num}.txt", "w") as file:
    #     loss_idx = np.nanargmin(train_history.history['val_loss'])
    #     file.write("Loss:\n")
    #     file.write(f"{train_history.history['val_loss'][loss_idx]}\n")
    #     acc = 1
    #     file.write("Accuracy:\n")
    #     for letter_idx in range(1, 13):
    #         acc *= train_history.history[f"val_label{letter_idx}_accuracy"][loss_idx]
    #     file.write(f"{acc}\n")
    #
    # plot(train_history.history, version_num)
    #
    # run.finish()
    # tf.keras.backend.clear_session()


def main():
    train(version_num=1, batch_size=64)


if __name__ == "__main__":
    main()
