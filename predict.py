from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.applications.imagenet_utils import decode_predictions
import config
import os

test_gen = ImageDataGenerator(rescale=1./255,fill_mode='nearest')

test_generator = test_gen.flow_from_directory(
    "data/tmp",
    config.image_size,
    shuffle=False,
    batch_size=1,
    class_mode='binary',
    )

model = load_model('first_try.h5')

pred = model.predict_generator(test_generator,steps = 200)

#np.savetxt('predict.csv', pred, delimiter=',')
#pred = np.argmax(pred, axis=1)
with open('result_of_dogs_in_validation.txt','w') as f:
    for i in range(len(pred)):
        f.write("%s %f\n" % (test_generator.filenames[i], pred[i]))
#print(classification_report(testLabels.argmax(axis=1), predIdxs,
#	target_names=lb.classes_))

'''
predicted_class_indices = np.argmax(pred, axis=1)
filenames = test_generator.filenames
labels = (train_generator.class_indices)
label = dict((v,k) for k,v in labels.items())

# 建立代码标签与真实标签的关系
predictions = [label[i] for i in predicted_class_indices]

#建立预测结果和文件名之间的关系
filenames = test_generator.filenames
for idx in range(len(filenames )):
    print('predict  %d' % (int(predictions[idx])))
    print('title    %s' % filenames[idx])
    print('')
'''