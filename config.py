from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

image_size = (250, 250)
input_shape = (250, 250, 3)
whether_to_generator = True
train_split_proportion = 0.2
steps_per_epoch = 100
epochs = 10
