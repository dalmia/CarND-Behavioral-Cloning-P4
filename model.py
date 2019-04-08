import csv
import cv2
import numpy as np
from os.path import join, exists
from os import makedirs
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

root = 'data'
image_path = join(root, 'IMG')
log_path = join(root, 'driving_log.csv')

NUM_EPOCHS = 7
BATCH_SIZE = 4
LR =  1e-4

print('Creating dataset...')
with open(log_path, 'r') as f:
	reader = csv.reader(f, delimiter=',')
	lines = []
	for line in reader:
		lines.append(line)

data_save_root = 'cache'
makedirs(data_save_root, exist_ok=True)
images_save_path = join(data_save_root, 'images.npy')
targets_save_path = join(data_save_root, 'targets.npy')

if not exists(images_save_path) or not exists(targets_save_path):
	images = []
	targets = []
	for line in lines[1:]:
		line = [_.strip() for _ in line]
		center_image_name = line[0].split('/')[-1]
		center_image_path = join(image_path, center_image_name)
		image = cv2.imread(center_image_path)
		images.append(image)

		steering_angle = float(line[3])
		targets.append(steering_angle)

	X_train = np.array(images)
	y_train = np.array(targets)
	print('Saving data to {}'.format(data_save_root))

	np.save(images_save_path, X_train)
	np.save(targets_save_path, y_train)

else:
	print('Loading cached data from {}'.format(data_save_root))
	X_train = np.load(images_save_path)
	y_train = np.load(targets_save_path)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

print('Creating model...')
# model = Sequential()
# model.add(Lambda(lambda x: x / 255., input_shape=(160, 320, 3)))
# model.add(Flatten())
# model.add(Dense(1))
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(160, 320, 3))
out = base_model.output

out = Flatten()(out)
out = Dense(4096, activation='relu')(out)
out = Dense(4096, activation='relu')(out)
out = Dense(1)(out)

model = Model(base_model.input, out)
optimizer = Adam(lr=LR)
model.compile(loss='mse', optimizer=optimizer)

print('Training...')
model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), 
					steps_per_epoch=len(X_train) / BATCH_SIZE, epochs=NUM_EPOCHS)

print('Done')
model.save('model.h5')

