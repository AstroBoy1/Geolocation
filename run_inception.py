'''
Copyright 2017 TensorFlow Authors and Kent Sommer
Edited by Michael Omori

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
from keras import backend as K
from inception import inception_v4
import numpy as np
import cv2
from keras.layers import Dense
import os
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# If you want to use a GPU set its index here
os.environ['CUDA_VISIBLE_DEVICES'] = ''


# This function comes from Google's ImageNet Preprocessing Script
def central_crop(image, central_fraction):
    """Crop the central region of the image.
	Remove the outer parts of an image but retain the central region of the image
	along each dimension. If we specify central_fraction = 0.5, this function
	returns the region marked with "X" in the below diagram.
	   --------
	  |        |
	  |  XXXX  |
	  |  XXXX  |
	  |        |   where "X" is the central 50% of the image.
	   --------
	Args:
	image: 3-D array of shape [height, width, depth]
	central_fraction: float (0, 1], fraction of size to crop
	Raises:
	ValueError: if central_crop_fraction is not within (0, 1].
	Returns:
	3-D array
	"""
    if central_fraction <= 0.0 or central_fraction > 1.0:
        raise ValueError('central_fraction must be within (0, 1]')
    if central_fraction == 1.0:
        return image

    img_shape = image.shape
    depth = img_shape[2]
    fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
    bbox_h_start = int(np.divide(img_shape[0], fraction_offset))
    bbox_w_start = int(np.divide(img_shape[1], fraction_offset))

    bbox_h_size = int(img_shape[0] - bbox_h_start * 2)
    bbox_w_size = int(img_shape[1] - bbox_w_start * 2)

    image = image[bbox_h_start:bbox_h_start + bbox_h_size, bbox_w_start:bbox_w_start + bbox_w_size]
    return image


def get_processed_image(img_path):
    # Load image and convert from BGR to RGB
    im = np.asarray(cv2.imread(img_path))[:, :, ::-1]
    im = central_crop(im, 0.875)
    im = cv2.resize(im, (299, 299))
    im = inception_v4.preprocess_input(im)
    if K.image_data_format() == "channels_first":
        im = np.transpose(im, (2, 0, 1))
        im = im.reshape(-1, 3, 299, 299)
    else:
        im = im.reshape(-1, 299, 299, 3)
    return im


if __name__ == "__main__":
    # https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
    # Use better and more images, at least take the ones without anything
    """TODO: Change to fit on both latitude and longitude
    1. Split the data
    2. Use a generator
    3. Train
    4. Predict
    5. Evaluate"""
    # Create model and load pre-trained weights
    model = inception_v4.create_model(weights='imagenet', include_top=False)

    # Freeze the inception base, going to just train the dense network
    for i in range(0, len(model.layers) - 1):
        model.layers[i].trainable = False

    model.compile(optimizer='rmsprop', loss='mse')

    df = pd.read_csv("photo_metadata.csv", nrows=100)

    train_df = df[:64]
    validtion_df = df[64:80]
    test_df = df[80:]

    datagen = ImageDataGenerator(rescale=1. / 255.)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory="images/",
        x_col="id",
        y_col="latitude",
        has_ext=False,
        batch_size=2,
        seed=42,
        shuffle=True,
        class_mode="other",
        target_size=(299, 299))

    valid_generator = datagen.flow_from_dataframe(
        dataframe=validtion_df,
        directory="images/",
        x_col="id",
        y_col="latitude",
        has_ext=False,
        batch_size=2,
        seed=42,
        shuffle=True,
        class_mode="other",
        target_size=(299, 299))

    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory="images/",
        x_col="id",
        y_col=None,
        has_ext=False,
        batch_size=32,
        seed=42,
        shuffle=False,
        class_mode=None,
        target_size=(299, 299))

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
    history = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator, validation_steps=STEP_SIZE_VALID, epochs=5)

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation loss')
    plt.title('Model Loss')
    plt.ylabel('RMSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    test_generator.reset()
    pred = model.predict_generator(test_generator, verbose=1)

    with open("history", 'wb') as f:
        pickle.dump(history, f)

    # Open Class labels dictionary. (human readable label given ID)
    # classes = eval(open('validation_utils/class_names.txt', 'r').read())

    # Load test image!
    # img_path = 'elephant.jpg'
    # img = get_processed_image(img_path)

    # Run prediction on test image
    # preds = model.predict(img)
    # print("Class is: " + classes[np.argmax(preds) - 1])
    # print("Certainty is: " + str(preds[0][np.argmax(preds)]))
