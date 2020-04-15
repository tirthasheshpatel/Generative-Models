import os
import tarfile

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
import keras
from keras import backend as K
import gan_utils.download_utils as download_utils
import gan_utils.tqdm_utils as tqdm_utils
import gan_utils.keras_utils as keras_utils

IMG_SIZE = int(os.getenv('IMG_SIZE'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))

def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def image_center_crop(img):
    """
    Makes a square center crop of an img, which is a [h, w, 3] numpy array.
    Returns [min(h, w), min(h, w), 3] output with same width and height.
    For cropping use numpy slicing.
    """
    h, w, c = img.shape
    h_crop = min(h,w)
    cropped_img = img[(h//2-h_crop//2):(h//2+h_crop//2), (w//2-h_crop//2):(w//2+h_crop//2), :]### YOUR CODE HERE

    # checks for errors
    # assert cropped_img.shape == (min(h, w), min(h, w), c), "error in image_center_crop!"

    return cropped_img

def prepare_raw_bytes_for_model(raw_bytes, normalize_for_model=False):
    img = decode_image_from_raw_bytes(raw_bytes)  # decode image raw bytes to matrix
    img = image_center_crop(img)  # take squared center crop
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # resize for our model
    img = img.astype("float32")  # prepare for normalization
    if normalize_for_model:
        img = keras.applications.inception_v3.preprocess_input(img)  # normalize for model
    return img

# reads bytes directly from tar by filename (slow, but ok for testing, takes ~6 sec)
def read_raw_from_tar(tar_fn, fn):
    with tarfile.open(tar_fn) as f:
        m = f.getmember(fn)
        return f.extractfile(m).read()

def get_all_filenames(tar_fn):
    with tarfile.open(tar_fn) as f:
        return [m.name for m in f.getmembers() if m.isfile()]

def raw_generator_with_label_from_tar(tar_fn, files, labels):
    label_by_fn = dict(zip(files, labels))
    with tarfile.open(tar_fn) as f:
        while True:
            m = f.next()
            if m is None:
                break
            if m.name in label_by_fn:
                yield f.extractfile(m).read(), label_by_fn[m.name]

def batch_generator(items, batch_size):
    batch = []
    i = 0
    for item in items:
        batch.append(item)
        i = i + 1
        if i == batch_size:
            i = 0
            yield batch
            batch = []
    yield batch

def train_generator(files, labels):
    while True:
        for batch in batch_generator(raw_generator_with_label_from_tar("data/102flowers.tgz", files, labels), BATCH_SIZE):
            batch_images = []
            for raw, label in batch:
                img = prepare_raw_bytes_for_model(raw)
                batch_images.append(img)

            batch_images = np.stack(batch_images, axis=0)
            yield batch_images, np.ones((batch_images.shape[0], 1))
