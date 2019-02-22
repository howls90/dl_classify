from keras.datasets import mnist
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam
import numpy as np
from keras.preprocessing import image
from PIL import Image
from glob import glob
import os
import cv2
import sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
import argparse
import random
import urllib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

target = 'target.png'
TRAIN_DIR = 'train'
TEST_DIR = 'test'
NEG_DIR = 'neg'
VAL_DIR = 'val'
TARG_DIR = 'targets'
# Create samples
background_num = 100
train_create = 10
test_create = 1
validation_create = 1
# Training
train_samples = background_num * train_create
test_samples = background_num * test_create
validation_samples = background_num * validation_create
# Hyperparameter
img_width = 28
img_height = 28
targ_width = 14
targ_height = 14
EPOCHS = 25
BATCH_SIZE = 32
LR = 1e-3
BIAS = tf.constant_initializer(0.01)

if not os.path.exists(TRAIN_DIR):
    os.rmdir( TRAIN_DIR )
    os.makedirs(TRAIN_DIR)

if not os.path.exists(TEST_DIR):
    os.rmdir( TEST_DIR )
    os.makedirs(TEST_DIR)

if not os.path.exists(VAL_DIR):
    os.rmdir( VAL_DIR )
    os.makedirs(VAL_DIR)

class CNN():
    def __self__():
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.negative = None


    def create_background(self):
        """
        Create background with noise
        """
        for i in range(0,background_num):
            background = np.random.random((img_width,img_height))
            plt.imsave(NEG_DIR+"/background.png", background)
            targ = Image.open(target)
            targ.thumbnail((targ_width, targ_height), Image.ANTIALIAS)
            background = Image.open(NEG_DIR+"/background.png")

            background.paste(targ,(0+random.randrange(img_width-targ.size[0]),0+random.randrange(img_height-targ.size[1])), targ)
            background.save(TARG_DIR+'/'+str(i)+'.png', quality=95)
        print("Backgrounds created succesfully!!!")


    def read_img(self):
        """
        Read images and process it
        """
        X_train = [TRAIN_DIR+'/'+i for i in os.listdir(TRAIN_DIR)]
        X_test = [TEST_DIR+'/'+i for i in os.listdir(TEST_DIR)]
        X_val = [VAL_DIR+'/'+i for i in os.listdir(VAL_DIR)]
        ng = [NEG_DIR+'/'+i for i in os.listdir(NEG_DIR)]

        X_train = X_train[:train_samples] + ng[:train_samples]
        X_test = X_test[:test_samples] + ng[train_samples:train_samples+test_samples]
        X_val = X_val[:validation_samples] + ng[train_samples+test_samples:train_samples+test_samples+validation_samples]

        # print("Training samples: "+str(len(X_train)))
        # print("Testing samples: "+str(len(X_test)))
        # print("validations samples: "+str(len(X_val)))

        random.shuffle(X_train)
        random.shuffle(X_test)
        random.shuffle(X_val)

        self.X_train, self.y_train = self.prepare_data(X_train)
        self.X_test, self.y_test = self.prepare_data(X_test)
        self.X_val, self.y_val = self.prepare_data(X_val)

    def create_images(self):
        """
        Create Images from target 
        """
        self.create_background()

        datagen = ImageDataGenerator(
                rotation_range=90,
                width_shift_range=0.2,
                height_shift_range=0.2,
                rescale = 1./255,
                shear_range=40,
                zoom_range=[1,3.5],
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest')

        j = 0
        for filename in glob(TARG_DIR+'/*'):
            print("Step: [{}/{}]".format(j,len(glob(TARG_DIR+'/*'))))
            img = load_img(filename)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            i = 0
            for batch in datagen.flow(x, batch_size=test_create, save_to_dir=TEST_DIR, save_prefix="{}_{}".format(j,i), save_format='jpeg'):
                i += 1
                if i == test_create:
                    break

            i = 0
            for batch in datagen.flow(x, batch_size=train_create, save_to_dir=TRAIN_DIR, save_prefix="{}_{}".format(j,i), save_format='jpeg'):
                i += 1
                if i == train_create:
                    break

            i = 0
            for batch in datagen.flow(x, batch_size=validation_create, save_to_dir=VAL_DIR, save_prefix="{}_{}".format(j,i), save_format='jpeg'):
                i += 1
                if i == validation_create:
                    break

            j = j + 1

    def prepare_data(self, positives):
        """
        Returns two arrays: 
            x is an array of resized images
            y is an array of labels
        """
        x = [] # images as arrays
        y = [] # labels

        for image in positives:
            _img = cv2.resize(cv2.imread(image), (img_width,img_height), interpolation=cv2.INTER_CUBIC)
            normalizedImg = np.zeros((img_width, img_height))
            normalizedImg = cv2.normalize(_img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
            x.append(_img)
                
            if image.split('/')[0] == TRAIN_DIR:
                y.append([1,0])
            elif image.split('/')[0] == NEG_DIR:
                y.append([0,1])
            elif image.split('/')[0] == TEST_DIR:
                y.append([1,0])
            elif image.split('/')[0] == VAL_DIR:
                y.append([1,0])

        x = np.array(x)
        y = np.array(y)
        return x, y

    def load_model(self):
        """
        Model definition
        """
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(32, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))
        
        self.model.add(Conv2D(64, kernel_size=3, activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu', kernel_initializer='random_uniform', bias_initializer=BIAS ))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax', kernel_initializer="random_uniform", bias_initializer=BIAS))
        opt = optimizer=Adam(lr=LR, decay=LR/EPOCHS)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    def training(self):
        """
        Train the nn
        """
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
        self.model.save_weights('weights.h5')


    def testing(self):
        """
        Test acuracy of the nn
        """
        self.model.load_weights('weights.h5')
        predict = np.argmax(self.model.predict(self.X_val), axis=1)
        validation = np.argmax(self.y_val, axis=1)
        pos = 0
        neg = 0
        for i in range(len(predict)):
            if predict[i] == validation[i]:
                pos = pos+1
            else:
                neg = neg+1
        print('Positives: '+str(pos))
        print('Negative: '+str(neg))

    def predict(self, img):
        """
        Prediction
        """
        x = []
        self.model.load_weights('weights.h5')
        x.append(cv2.resize(cv2.imread(img), (img_width,img_height), interpolation=cv2.INTER_CUBIC))
        x = np.array(x)
        if np.argmax(self.model.predict(x), axis=1)[0] == 0:
            print('Yes')
        else:
            print('No')
        plt.imshow(x[0])
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object classification.')
    parser.add_argument('--samples', help='train the cnn', action='store_true')
    parser.add_argument('--train', help='train the cnn', action='store_true')
    parser.add_argument('--test', help='test the cnn', action='store_true')
    parser.add_argument('--predict', help='predict if obj in img')
    args = parser.parse_args()

    cnn = CNN()
    
    if args.train:
        cnn.read_img()
        cnn.load_model()
        cnn.training()
    elif args.test:
        cnn.read_img()
        cnn.load_model()
        cnn.testing()
    elif args.samples:
        cnn.create_images()
    elif args.predict:
        cnn.read_img()
        cnn.load_model()
        cnn.predict(args.predict)
