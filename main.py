from typing import Optional, List, Tuple

import tensorflow as tf
import pathlib

from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.client.session import InteractiveSession

"""

AUTOTUNE = tf.data.experimental.AUTOTUNE
data_root = pathlib.Path("data/test/input")
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_paths = [str(path) for path in list(data_root.glob('*/*'))]
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]
image_count = len(all_image_paths)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
BATCH_SIZE = 32

# Установка размера буфера перемешивания, равного набору данных, гарантирует
# полное перемешивание данных.
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
ds = ds.prefetch(buffer_size=AUTOTUNE)

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable = False


def change_range(image, label):
    return 2 * image - 1, label


keras_ds = ds.map(change_range)
image_batch, label_batch = next(iter(keras_ds))

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_names), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
model.fit(ds, epochs=500, steps_per_epoch=1800)
"""


class Model:

    def __init__(self, path: str = "data/test/input", model_ai: Optional[tf.keras.Sequential] = None,
                 input_shape_mobile_net_v2: Tuple = (192, 192, 3)):
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.data_root = pathlib.Path(path)
        self.label_names = sorted(item.name for item in self.data_root.glob('*/') if item.is_dir())
        self.label_to_index = dict((name, index) for index, name in enumerate(self.label_names))
        self.all_image_paths = [str(path) for path in list(self.data_root.glob('*/*'))]
        self.all_image_labels = [self.label_to_index[pathlib.Path(path).parent.name]
                                 for path in self.all_image_paths]
        self.image_count = len(self.all_image_paths)
        self.mobile_net = tf.keras.applications.MobileNetV2(input_shape=input_shape_mobile_net_v2, include_top=False)
        self.mobile_net.trainable = False
        if model_ai is None:
            self.model = tf.keras.Sequential([
                self.mobile_net,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(len(self.label_names), activation='softmax')
            ])
        else:
            self.model = model_ai
        self.path_ds = tf.data.Dataset.from_tensor_slices(self.all_image_paths)
        self.image_ds = self.path_ds.map(self.load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        self.label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(self.all_image_labels, tf.int64))
        self.image_label_ds = tf.data.Dataset.zip((self.image_ds, self.label_ds))
        self.BATCH_SIZE = 32
        ds1 = self.image_label_ds.shuffle(buffer_size=self.image_count)
        ds1 = ds1.repeat()
        ds1 = ds1.batch(self.BATCH_SIZE)
        self.ds = ds1.prefetch(buffer_size=self.AUTOTUNE)

    def compile(self, optimizer: Optional[tf.keras.Sequential] = None, loss: Optional[str] = None,
                metrics: Optional[List[str]] = None):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam()
        if loss is None:
            loss = 'sparse_categorical_crossentropy'
        if metrics is None:
            metrics = ["accuracy"]

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)
        return self

    def fit(self, epochs: int = 500, steps_per_epoch: int = 1800):
        self.model.fit(self.ds, epochs=epochs, steps_per_epoch=steps_per_epoch)
        return self

    def cache_memory(self):
        ds = self.image_label_ds.cache()
        ds = ds.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=self.image_count)
        )
        self.ds = ds.batch(self.BATCH_SIZE).prefetch(buffer_size=self.AUTOTUNE)
        return self

    def cache_file(self, path: str = './cache.tf-data'):
        ds = self.image_label_ds.cache(filename=path)
        ds = ds.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=self.image_count))
        self.ds = ds.batch(self.BATCH_SIZE).prefetch(1)
        return self

    def evaluate(self, x, y, verbose: int = 2):
        self.model.evaluate(x, y, verbose=verbose)
        return self

    def save_weights(self, path: str = "./checkpoints/Model"):
        self.model.save_weights(path)
        return self

    def load_weights(self, path: str = "./checkpoints/Model"):
        self.model.load_weights(path)
        return self

    def save_model(self, file: str = "model/ai.h5"):
        self.model.save(file)
        return self

    def load_model(self, file: str = "model/ai.h5"):
        self.model = tf.keras.models.load_model(file)
        return self

    def summary(self):
        return self.model.summary()

    @staticmethod
    def change_range(image, label):
        return 2 * image - 1, label

    @staticmethod
    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [192, 192])
        image /= 255.0

        return image

    @staticmethod
    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        return Model.preprocess_image(image)


if __name__ == '__main__':
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    model = Model()
    model.compile().fit(epochs=50).save_model()
