from pyspark import SparkConf
from pyspark import SparkContext

sc.install_pypi_package("datetime", "https://pypi.org/simple")
sc.install_pypi_package("keras", "https://pypi.org/simple")
# sc.install_pypi_package("numpy", "https://pypi.org/simple") #comes default
sc.install_pypi_package("Pillow", "https://pypi.org/simple")
sc.install_pypi_package("boto3", "https://pypi.org/simple")

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
import keras.optimizers
import os
import numpy as np
from PIL import Image
from datetime import datetime
import boto3
from io import BytesIO

s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')
bucket = 'cs383finalproject'
my_bucket = s3_resource.Bucket('cs383finalproject')

def getPixelsFromS3(s3_client, bucket, key):
    file_obj = s3_client.get_object(Bucket = bucket, Key=key)

    file_contents = file_obj['Body'].read()
    pic = Image.open(BytesIO(file_contents))
    np_im = np.array(pic)
    return np_im

os.getcwd()
os.chdir('tmp') #for permission purposes


def sized_model(sample_size):
    melanoma_directory = "melanoma/"
    nevus_directory = "nevus/"
    seb_directory = "seborrheic_keratosis/"
    removalcount = 0
    resizecount = 0
    IMG_SIZE = (100, 100)
    SAMPLE_SIZE = sample_size
    valid_size = 100

    i = 0
    k = 0
    l = 0
    melobjects = my_bucket.objects.filter(Prefix="picscopy/melanoma/")
    nevobjects = my_bucket.objects.filter(Prefix="picscopy/nevus/")
    sebobjects = my_bucket.objects.filter(Prefix="picscopy/seborrheic_keratosis/")
    print("loading melanoma images...")
    mel_full_set = np.asarray(
        [getPixelsFromS3(s3_client, bucket, mel.key) for mel in melobjects if mel.key.endswith('.jpg')])
    print("loading nevus images...")
    nev_full_set = np.asarray(
        [getPixelsFromS3(s3_client, bucket, nev.key) for nev in nevobjects if nev.key.endswith('.jpg')])
    print("loading seb images...")
    seb_full_set = np.asarray(
        [getPixelsFromS3(s3_client, bucket, seb.key) for seb in sebobjects if seb.key.endswith('.jpg')])

    mel_train_set = mel_full_set[0:SAMPLE_SIZE]
    nev_train_set = nev_full_set[0:SAMPLE_SIZE]
    seb_train_set = seb_full_set[0:SAMPLE_SIZE]

    mel_valid_set = mel_full_set[-valid_size:]
    nev_valid_set = nev_full_set[-valid_size:]
    seb_valid_set = seb_full_set[-valid_size:]

    # generate X and Y (inputs and labels).
    x_train = np.concatenate([mel_train_set, nev_train_set, seb_train_set])
    labels_train = np.asarray(
        [0 for _ in range(SAMPLE_SIZE)] + [1 for _ in range(SAMPLE_SIZE)] + [2 for _ in range(SAMPLE_SIZE)])
    labels_train = keras.utils.to_categorical(labels_train)

    x_valid = np.concatenate([mel_valid_set, nev_valid_set, seb_valid_set])
    labels_valid = np.asarray(
        [0 for _ in range(valid_size)] + [1 for _ in range(valid_size)] + [2 for _ in range(valid_size)])
    labels_valid = keras.utils.to_categorical(labels_valid)
    fc_layer_size = 256
    img_size = IMG_SIZE

    conv_inputs = keras.Input(shape=(img_size[1], img_size[0], 3), name='ani_image')
    conv_layer = keras.layers.Conv2D(128, kernel_size=3, activation='relu')(conv_inputs)
    conv_layer = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_layer)

    conv_layer = keras.layers.Conv2D(128, kernel_size=3, activation='relu')(conv_layer)
    conv_layer = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_layer)

    conv_x = keras.layers.Flatten(name='flattened_features')(conv_layer)  # turn image to vector.

    conv_x = keras.layers.Dense(fc_layer_size, activation='relu', name='first_layer')(conv_x)
    conv_x = keras.layers.Dense(fc_layer_size, activation='relu', name='second_layer')(conv_x)
    conv_outputs = keras.layers.Dense(3, activation='softmax', name='class')(conv_x)

    conv_model = keras.Model(inputs=conv_inputs, outputs=conv_outputs)

    customAdam = keras.optimizers.Adam(lr=1e-6)
    conv_model.compile(optimizer=customAdam,  # Optimizer
                       # Loss function to minimize
                       loss="categorical_crossentropy",
                       # List of metrics to monitor
                       metrics=["accuracy"])

    history = conv_model.fit(x_train,
                             labels_train,
                             batch_size=32,
                             shuffle=True,
                             epochs=150,
                             steps_per_epoch=10,
                             validation_data=(x_valid, labels_valid))

    preds = conv_model.predict(x_valid)

    scores = conv_model.evaluate(x_valid, labels_valid, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    my_bucket.download_file('skin-lesions-model-output.txt', 'skin-lesions-model-output.txt')

    with open('skin-lesions-model-output.txt', 'a') as output:
        output.write('Model with n=%d, image size of %d x %d, Accuracy of %.2f %% recorded at %s\n' % (
        SAMPLE_SIZE, IMG_SIZE[0], IMG_SIZE[1], (scores[1] * 100), datetime.now()))
    my_bucket.upload_file('skin-lesions-model-output.txt', 'skin-lesions-model-output.txt')

sized_model(1000)