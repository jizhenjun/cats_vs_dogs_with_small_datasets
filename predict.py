from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.applications.imagenet_utils import decode_predictions
import config
import os

test_gen = ImageDataGenerator(rescale=1./255,fill_mode='nearest')

test_generator = test_gen.flow_from_directory(
    "data/test",
    config.image_size,
    shuffle=False,
    batch_size=1,
    class_mode='binary',
    )

model = load_model('models/test.h5')

pred = model.predict_generator(test_generator,steps = 12500)
y=[]
for i in range(len(pred)):
	y.append(0)
for i in range(len(pred)):
	index = test_generator.filenames[i].split('.',1)[0]
	#print(index)
	#index = index.split('.',1)[0]
	index = index.split('\\',1)[1]
	index = int(index)

	pred[i] = min(pred[i],0.995)
	pred[i] = max(pred[i],0.005)
	y[index - 1] = pred[i]

np.savetxt('results/result_of_test.csv', y, delimiter=',')
#pred = np.argmax(pred, axis=1)
'''
with open('result_of_dogs_in_validation.txt','w') as f:
    for i in range(len(pred)):
        f.write("%s %f\n" % (test_generator.filenames[i], pred[i]))
'''