'''
WARNING: 

It is a simple classifier and it is NOT optimized for best results!
Only use for learning purposes. Can be used on any dataset for basic beginner level image classification
'''
#import dependencies
import numpy as numpy
from keras.models import Sequential
from keras.layers import Dropout, Flatten,Dense,Conv2D,ZeroPadding2D,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
#load data
l,b = 64,64
generator = ImageDataGenerator(rescale = 1./255)#for rescaling 0-255 to 0-1
final_train_data=generator.flow_from_directory('dataset/train',
	target_size=(l,b),#add required image path
	batch_size=16,
	class_mode='binary')

final_test_data=generator.flow_from_directory('dataset/test',
	target_size=(l,b),
	batch_size=32,
	class_mode='binary')
#conv1
model=Sequential() #instantiate model
model.add(Conv2D(32,(3,3),input_shape=(l,b, 3), activation='relu' ))##BAD PROGRAMMING PRACTICE(for understanding purpose)
model.add(MaxPooling2D(pool_size=(2,2)))                                                      
#conv2
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#conv3
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#create first fully connected layer
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
#create second fully connected layer
model.add(Dense(1,activation='relu')) ##final classification layer


model.compile(loss='binary_crossentropy',
	optimizer='rmsprop',
	metrics=['accuracy'])

epochs=20
n_train=8000
n_test=2000

model.fit_generator(final_train_data,
	steps_per_epoch = n_train, #samples per epoch
	epochs = epochs,
	validation_data = final_test_data,
	validation_steps = n_test)


model.save_weights('models/ClassifiesCNN.h5')
# for loading use model.load_weights('models_trained/ClassifiesCNN.h5')
