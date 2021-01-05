from typing import Optional, List, Tuple, Any

import numpy
import tensorflow as tf
import pathlib
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.client.session import InteractiveSession


class Flowers:

    def __init__(self, path: str = "data/test/input", model_ai: Optional[tf.keras.Sequential] = None,
                 input_shape_mobile_net_v2: Tuple = (192, 192, 3)):
        """

        :param path:
        :param model_ai:
        :param input_shape_mobile_net_v2:
        """
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
        print()
        self.BATCH_SIZE = 32
        ds1 = self.image_label_ds.shuffle(buffer_size=self.image_count)
        ds1 = ds1.repeat()
        ds1 = ds1.batch(self.BATCH_SIZE)
        self.ds = ds1.prefetch(buffer_size=self.AUTOTUNE)

    def compile(self, optimizer: Optional[tf.keras.Sequential] = None, loss: Optional[str] = None,
                metrics: Optional[List[str]] = None):
        """

        :param optimizer:
        :param loss:
        :param metrics:
        :return:
        """
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
        """

        :param epochs:
        :param steps_per_epoch:
        :return:
        """
        self.model.fit(self.ds, epochs=epochs, steps_per_epoch=steps_per_epoch)
        return self

    def cache_memory(self):
        """

        :return:
        """
        ds = self.image_label_ds.cache()
        ds = ds.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=self.image_count)
        )
        self.ds = ds.batch(self.BATCH_SIZE).prefetch(buffer_size=self.AUTOTUNE)
        return self

    def cache_file(self, path: str = './cache.tf-data'):
        """

        :param path:
        :return:
        """
        ds = self.image_label_ds.cache(filename=path)
        ds = ds.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=self.image_count))
        self.ds = ds.batch(self.BATCH_SIZE).prefetch(1)
        return self

    def evaluate(self, x, y, verbose: int = 2):
        """

        :param x:
        :param y:
        :param verbose:
        :return:
        """
        self.model.evaluate(x, y, verbose=verbose)
        return self

    def save_weights(self, path: str = "./checkpoints/Model"):
        """

        :param path:
        :return:
        """
        self.model.save_weights(path)
        return self

    def load_weights(self, path: str = "./checkpoints/Model"):
        """

        :param path:
        :return:
        """
        self.model.load_weights(path)
        return self

    def save_model(self, file: str = "model/ai.h5"):
        """

        :param file:
        :return:
        """
        self.model.save(file)
        return self

    def load_model(self, file: str = "model/ai.h5"):
        """

        :param file:
        :return:
        """
        self.model = tf.keras.models.load_model(file)
        return self

    def summary(self):
        """

        :return:
        """
        return self.model.summary()

    def get_prediction(self, x: Any = None,
                       path: Optional[List[str]] = None,
                       batch_size: Any = None,
                       verbose: int = 0,
                       steps: Any = None,
                       callbacks=None,
                       max_queue_size: int = 10,
                       workers: int = 1,
                       use_multiprocessing: bool = False):
        """
        if x is None then image form path = x
        :param x:
        :param path:
        :param batch_size:
        :param verbose:
        :param steps:
        :param callbacks:
        :param max_queue_size:
        :param workers:
        :param use_multiprocessing:
        :return:
        """
        if x is None:
            path_t = tf.data.Dataset.from_tensor_slices(path)
            x = path_t.map(self.load_and_preprocess_image, num_parallel_calls=model.AUTOTUNE).batch(32)
        result = self.model.predict(x, batch_size,
                                    verbose,
                                    steps,
                                    callbacks,
                                    max_queue_size,
                                    workers,
                                    use_multiprocessing)
        result_list = []
        for i in result:
            result_list.append(self.label_names[numpy.argmax(i)])
        return result_list

    @staticmethod
    def change_range(image, label):
        """

        :param image:
        :param label:
        :return:
        """
        return 2 * image - 1, label

    @staticmethod
    def preprocess_image(image):
        """

        :param image:
        :return:
        """
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [192, 192])
        image /= 255.0

        return image

    @staticmethod
    def load_and_preprocess_image(path):
        """

        :param path:
        :return:
        """
        image = tf.io.read_file(path)
        return Flowers.preprocess_image(image)


if __name__ == '__main__':
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    model = Flowers()
    print(model.load_model().get_prediction(path=[ "data/test/input/melon/images (10).jpeg", "data/test/input/melon"
                                                                                             "/images (1).jpeg",
                                                   "data/test/input/roses/24781114_bc83aa811e_n.jpg"]))
