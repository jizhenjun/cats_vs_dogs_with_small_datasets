from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.applications.imagenet_utils import decode_predictions
import config

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

def predict_cnnsvm(k,pca=None):#测试批数，共用数据32*k
    s,al=0,0
    for i in range(k):
        res=validation_generator.next()
        x,y=res[0],res[1]
        x_temp=model.predict(x)
        if pca!=None:
            y_temp=clf.predict(pca.transform(x_temp))
        else:
            y_temp = clf.predict(x_temp)
        s+=np.sum(y_temp==y)
        al+=len(y)
    return s*1.0/al
	
model = load_model('model/k_5_dnn_250_250.h5')
model = Model(inputs=model.input,outputs=model.layers[-2].output)

from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np
clf=SVC()

X=np.ones((0,1024))
Y=np.array([])
for i in range(50):
    res=train_generator.next()
    x,y=res[0],res[1]
    x_temp=model.predict(x)
    X=np.row_stack((X,x_temp))
    Y=np.append(Y,y)
    print("%d inserted!"%(i*32+32))
print(X.shape)
print(Y.shape)

print("no use pca:")
clf.fit(X,Y)
#clf.save('test.h5')
for _ in range(5):
    pre=predict_cnnsvm(25)
    print("correct_rate:%.3f"%(pre))
'''
print("use pca:")
pca = PCA(n_components=10)
pca.fit(X)
X_new = pca.transform(X)
clf.fit(X_new,Y)
for _ in range(5):
    pre=predict_cnnsvm(20,pca)
    print("correct_rate:%.3f"%(pre))
'''