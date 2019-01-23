from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

image_size = (50, 50)
input_shape = (50, 50, 3)
whether_to_generator = False
train_split_proportion = 0.2
steps_per_epoch = 2000
epochs = 50
