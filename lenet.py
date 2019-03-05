# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Dropout, BatchNormalization
from keras import backend as K
from glob import glob
from PIL import Image,ImageEnhance

import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import to_categorical
# from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import imutils
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
shape = 28
target = 'target.png'

if not os.path.exists('images/'):
	os.makedirs('images/')

if not os.path.exists('images/ring'):
	os.makedirs('images/ring')

if not os.path.exists('images/not_ring'):
	os.makedirs('images/not_ring')

 
class LeNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)
 
		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))
		model.add(Dropout(0.25))

		# model.add(Dense(250))
		# model.add(Activation("relu"))
		# model.add(Dropout(0.3))

		model.add(Dense(50))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))
 
		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("sigmoid"))
 
		# return the constructed network architecture
		return model

def train():
	print("[INFO] loading images...")
	data = []
	labels = []
	
	# grab the image paths and randomly shuffle them
	imagePaths = sorted(list(paths.list_images('images/')))
	random.seed(42)
	random.shuffle(imagePaths)

	# loop over the input images
	for imagePath in imagePaths:
		# load the image, pre-process it, and store it in the data list
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (shape, shape))
		image = img_to_array(image)
		data.append(image)
	
		# extract the class label from the image path and update the
		# labels list
		label = imagePath.split(os.path.sep)[-2]
		label = 1 if label == "ring" else 0
		labels.append(label)

	# scale the raw pixel intensities to the range [0, 1]
	data = np.array(data, dtype="float") / 255.0
	labels = np.array(labels)
	
	# partition the data into training and testing splits using 75% of
	# the data for training and the remaining 25% for testing
	(trainX, testX, trainY, testY) = train_test_split(data[:2000],
		labels[:2000], test_size=0.25, random_state=50)
	
	# convert the labels from integers to vectors
	trainY = to_categorical(trainY, num_classes=2)
	testY = to_categorical(testY, num_classes=2)

	# construct the image generator for data augmentation
	aug = ImageDataGenerator(rotation_range=30, 
								shear_range=0.2,
								vertical_flip=True, 
								brightness_range=[0.3, 1.3], 
								fill_mode="nearest")

	# initialize the model
	print("[INFO] compiling model...")
	model = LeNet.build(width=shape, height=shape, depth=1, classes=2)
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="binary_crossentropy", optimizer=opt,
		metrics=["accuracy"])
	
	# train the network
	print("[INFO] training network...")
	# H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	# 	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	# 	epochs=EPOCHS, verbose=1)
	H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=BS)
	
	# save the model to disk
	print("[INFO] serializing network...")
	model.save('lenet.model')

	# plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	N = EPOCHS
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy on Santa/Not Santa")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig('lenet.png')

def test():
	cap = cv2.VideoCapture(0)
	model = load_model('lenet.model')

	while 1:
		_, original = cap.read()
		ret, image = cap.read()
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		image = image.astype("float") / 255.0
		image = cv2.resize(image, (shape, shape))
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)

		# classify the input image
		(notSanta, santa) = model.predict(image)[0]

		# build the label
		label = "Ring" if santa > notSanta else "Not Ring"
		proba = santa if santa > notSanta else notSanta
		label = "{}: {:.2f}%".format(label, proba * 100)
		print(label)
		
		cv2.imshow('img',original)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
	cap.release()
	cv2.destroyAllWindows()

	
def samples():
	"""
	Create background with noise
	"""
	j = 0
	for filename in glob('images/not_ring/*'):
	# for j in range(1,int(args['samples'])):
		# filename = "images_change/not_ring/1_"+str(j)+".png"
		# background = np.random.random((shape,shape))
		# plt.imsave(filename, background)

		targ = Image.open(target)
		targ = targ.rotate(random.randrange(-45, 45))
		# targ.thumbnail((shape-10 shape-10), Image.ANTIALIAS)

		width, height = targ.size
		m = random.uniform(-0.2, 0.2)
		xshift = abs(m) * width
		new_width = width + int(round(xshift))
		targ = targ.transform((new_width, height), Image.AFFINE,
				(1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)

		background = Image.open(filename)
		background.thumbnail((shape, shape), Image.ANTIALIAS)
		print(targ.size)
		background.paste(targ,(0,random.randrange(shape-targ.size[1])), targ)
		background = ImageEnhance.Brightness(background).enhance(random.uniform(0.8, 1))
		background = ImageEnhance.Contrast(background).enhance(random.uniform(0.6, 1))
		background.save('images/ring/'+str(j)+'.png', quality=95)
		j += 1
	print("Backgrounds created succesfully!!!")
	# for j in range(1,10000):
	# 	filename = "images_change/not_ring/1_"+str(j)+".png"
	# 	background = np.random.random((shape,shape))
	# 	plt.imsave(filename, background)
	# 	# targ = Image.open(target)
	# 	# targ.thumbnail((shape, shape), Image.ANTIALIAS)
	# 	# background = Image.open(filename)
	# 	# background.thumbnail((shape, shape), Image.ANTIALIAS)

	# 	# background.paste(targ,(0,0), targ)
	# 	# background.save('images_change/not_ring/1_'+str(j)+'.png', quality=95)
	# print("Backgrounds created succesfully!!!")

if __name__ == "__main__":
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--cam", help="path to input image" , action='store_true')
	ap.add_argument("-s", "--samples", help="number of samples")
	args = vars(ap.parse_args())

	if args['cam']:
		test()
	elif args['samples']:
		samples()
	else:
		train()
	


	