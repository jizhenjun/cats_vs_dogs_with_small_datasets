from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
#import wechat_utils
import config
#wechat_utils.login()
if config.whether_to_generator:
    train_gen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    #zca_whitening=True,
    #vertical_flip=False,
    fill_mode='nearest',
    #validation_split=config.train_split_proportion
    )
else:
    train_gen = ImageDataGenerator(rescale=1./255, 
	#validation_split=config.train_split_proportion
	)
test_gen = ImageDataGenerator(rescale=1./255)
train_generator = train_gen.flow_from_directory(
    "data/train",
    config.image_size,
    shuffle=True,
    batch_size=32,
    class_mode = 'binary',
    #subset='training'
    )
validation_generator = test_gen.flow_from_directory(
    "data/validation",
    config.image_size,
    shuffle=True,
    batch_size=16,
    class_mode='binary',
    #subset='validation'
    )

def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = config.input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    model.summary()
    return model

def AlexNet_model():
    seed = 7  
    np.random.seed(seed) 
    model = Sequential()

    model.add(Conv2D(32,(5,5),strides=(1,1),padding='same', input_shape = config.input_shape,activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
    model.add(Conv2D(32,(4,4),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
    model.add(Conv2D(64,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))    
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  

    model.add(Flatten())  
    model.add(Dense(2048,activation='relu'))  
    model.add(Dropout(0.4))  
    model.add(Dense(1024,activation='relu'))  
    model.add(Dropout(0.4))  
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer='adam')
    # UNCOMMENT THIS TO VIEW THE ARCHITECTURE
    model.summary()
    
    return model

def dnn_model():
    seed = 2048 
    np.random.seed(seed) 
    model = Sequential()

    model.add(Conv2D(32, (5, 5), strides=1, input_shape = config.input_shape,activation='relu', padding='same',kernel_initializer='uniform'))
    model.add(Conv2D(32, (5, 5), strides=1, activation='relu', padding='same',kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5, 5), strides=1,activation='relu',padding='same',kernel_initializer='uniform'))
    model.add(Conv2D(64, (5, 5), strides=1,activation='relu',padding='same',kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), strides=1,activation='relu',padding='same',kernel_initializer='uniform'))
    model.add(Conv2D(128, (3, 3), strides=1,activation='relu',padding='same',kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), strides=1,activation='relu',padding='same',kernel_initializer='uniform'))
    model.add(Conv2D(128, (3, 3), strides=1,activation='relu',padding='same',kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.5))

    model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer='adam')
    # UNCOMMENT THIS TO VIEW THE ARCHITECTURE
    model.summary()
    
    return model

model = dnn_model()

from keras.callbacks import *

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)
checkpointer = ModelCheckpoint(filepath="./checkpoint.hdf5", verbose=1)

def scheduler(epoch):
	if epoch == 0:
		lr = K.get_value(model.optimizer.lr)
		K.set_value(model.optimizer.lr, lr * 10)
		print("lr changed to {}".format(lr * 10))
	return K.get_value(model.optimizer.lr)
	'''
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)
	'''

reduce_lr = LearningRateScheduler(scheduler)



model.fit_generator(
    train_generator,
    steps_per_epoch=config.steps_per_epoch,
    verbose=1,
    epochs=config.epochs,
    validation_data=validation_generator,
    callbacks = [tensorboard, checkpointer #, reduce_lr
    #wechat_utils.sendmessage(savelog=True,fexten='TEST')
    ],
    validation_steps = 25)
# always save your weights after training or during training

hist = model.save('test.h5')

#with open('log_sgd_big_32.txt','w') as f:
#    f.write(str(hist.History))


