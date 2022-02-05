import os
import numpy as np
from typing import List, Dict
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense

class Fer:
    dataset_path = "dataset"
    model_path = os.path.join("model", "best_fer.hdf5")
    input_shape = (48, 48, 1)

    def __init__(self):
        self.emotion_classes = self._get_classes()
        self.emotion_dict = {i:ec for i, ec in enumerate(self.emotion_classes)}

        self.model = self._prepare_model()


    def get_emotion_dict(self) -> Dict[int, str]:
        return self.emotion_dict


    def get_prediction(self, img) -> Dict[int, float]:
        img = np.reshape(img, (1, 48, 48, 1))
        prediction = self.model.predict(img)[0]
        prediction_dict = {i:p for i, p in enumerate(prediction)}
        return prediction_dict


    def _get_classes(self) -> List[str]:
        train_path = os.path.join(self.dataset_path, "train")
        emotion_classes = os.listdir(train_path)
        return emotion_classes


    def _get_model(self, input_shape, num_classes):
        model = tf.keras.models.Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape, data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        return model


    def _prepare_model(self):
        model = self._get_model(self.input_shape, len(self.emotion_classes))
        model.load_weights(self.model_path)
        return model