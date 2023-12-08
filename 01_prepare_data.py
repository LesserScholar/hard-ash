from typing import Any
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import shutil

shutil.rmtree('./mnist_split', ignore_errors=True)
for i in range(5):
    dataset = tfds.load('mnist', split='train', as_supervised=True, shuffle_files=True)
    dataset = dataset.filter(lambda x, y: (y // 2) == i).shuffle(1000)
    dataset.save(f'mnist_split/{i}')

# dataset = tfds.load('mnist', split='test', as_supervised=True, shuffle_files=True)
# dataset.save('resto_data/mnist_test')