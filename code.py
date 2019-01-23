from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import config

if config.whether_to_generator:
    train_gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=config.train_split_proportion
    )
else:
    train_gen = ImageDataGenerator(rescale=1./255, validation_split=config.train_split_proportion)
test_gen = ImageDataGenerator(rescale=1./255)
train_generator = train_gen.flow_from_directory(
    "data/train",
    config.image_size,
    shuffle=True,
    batch_size=32,
    class_mode = 'binary',
    #subset='training'
    )
test_generator = test_gen.flow_from_directory(
    "data/test",
    config.image_size,
    shuffle=False,
    batch_size=32,
    class_mode=None
    )
validation_generator = test_gen.flow_from_directory(
    'data/validation',
    config.image_size,
    shuffle=True,
    batch_size=32,
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
    model.add(Dense(64))
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

model.fit_generator(
    train_generator,
    steps_per_epoch=config.steps_per_epoch,
    verbose=1,
    epochs=config.epochs,
    validation_data=validation_generator,
    validation_steps = 1)
# always save your weights after training or during training
model.save_weights('first_try.h5')
