import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from random import choice, sample
import cv2
from imageio import imread
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras import regularizers
from keras.layers import Input, Embedding, LSTM, Dropout, BatchNormalization,Dense, concatenate, Flatten, Conv1D
from keras.optimizers import RMSprop, Adam
from keras_vggface.vggface import VGGFace
from glob import glob
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing import image
from keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D, Lambda, Reshape
from collections import defaultdict
from keras_vggface.utils import preprocess_input

## ===============


#loading the data
TRAIN_BASE = 'train'    #folder containing training data
families = sorted(os.listdir(TRAIN_BASE))
print('We have {} families in the dataset'.format(len(families)))

#all_images contains paths of images.
all_images = glob(TRAIN_BASE + "*/*/*/*.jpg")

#Splitting the data into train & validation set.
#Validation set includes family with name F09.
val_families = "F09"
train_images = [x for x in all_images if val_families not in x]
val_images = [x for x in all_images if val_families in x]





## ===============


ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

train_person_to_images_map = defaultdict(list)

for x in train_images:
    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

val_person_to_images_map = defaultdict(list)

for x in val_images:
    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

relationships = pd.read_csv('train_relationships.csv')
relationships = list(zip(relationships.p1.values, relationships.p2.values))
relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

train = [x for x in relationships if val_families not in x[0]]
val = [x for x in relationships if val_families in x[0]]


## ===============


#Facenet architecture will take image of size 160 x 160
IMG_SIZE_FN = 160
#Facenet architecture will take image of size 224 x 224
IMG_SIZE_VGG = 224

model_path = 'facenet_keras.h5'
facenet_model = load_model(model_path)

#We will train full network except the last 3 layers
for layer in facenet_model.layers[:-3]:
    layer.trainable = True

#We will train full network except the last 3 layers
vgg_model = VGGFace(model='resnet50', include_top=False)
for layer in vgg_model.layers[:-3]:
    layer.trainable = True



#this model takes four inputs
input_1 = Input(shape=(IMG_SIZE_FN, IMG_SIZE_FN, 3))        #facenet for Image 1
input_2 = Input(shape=(IMG_SIZE_FN, IMG_SIZE_FN, 3))        #facenet for image 2
input_3 = Input(shape=(IMG_SIZE_VGG, IMG_SIZE_VGG, 3))      #VGG for image 1
input_4 = Input(shape=(IMG_SIZE_VGG, IMG_SIZE_VGG, 3))      #VGG for image 2

vgg_1 = vgg_model(input_3)
vgg_2 = vgg_model(input_4)
fn_1 = facenet_model(input_1)
fn_2 = facenet_model(input_2)
    
x1 = Reshape((1, 1 ,128))(fn_1)   #reshaping image array for global max pool layer
x2 = Reshape((1, 1 ,128))(fn_2)
x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

#the below 4 lamda functions will calcluate the square of each input image
lambda_1 = Lambda(lambda tensor  : K.square(tensor))(fn_1) 
lambda_2 = Lambda(lambda tensor  : K.square(tensor))(fn_2)
lambda_3 = Lambda(lambda tensor  : K.square(tensor))(vgg_1)
lambda_4 = Lambda(lambda tensor  : K.square(tensor))(vgg_2)
    
added_facenet = Add()([x1, x2])     #this function will add two images image 1 image 2 given by facenet architecture
added_vgg = Add()([vgg_1, vgg_2])    #this function will add two images image 3 image 4 given by VGG architecture
subtract_fn = Subtract()([x1,x2])    #this function will subtract two images image 1 image 2 given by facenet architecture
subtract_vgg = Subtract()([vgg_1,vgg_2])   #this function will subtract two images image 3 image 4 given by VGG architecture
subtract_fn2 = Subtract()([x2,x1])    #this function will subtract two images image 2 image 1 given by facenet architecture
subtract_vgg2 = Subtract()([vgg_2,vgg_1])   #this function will subtract two images image 4 image 3 given by VGG architecture
prduct_fn1 = Multiply()([x1,x2])    #this function will multiply two images image 1 image 2 given by facenet architecture
prduct_vgg1 = Multiply()([vgg_1,vgg_2])   #this function will multiply two images image 3 image 4 given by VGG architecture
sqrt_fn1 = Add()([lambda_1,lambda_2])        # this function implements x1^2 + x2^2 where x1 and x2 are image by facenet
sqrt_vgg1 = Add()([lambda_3,lambda_4])       # this function implements vgg_1^2 + vgg_2^2 where vgg_1 and vgg_2 are image by VGG
sqrt_fn2 = Lambda(lambda tensor  : K.sign(tensor)*K.sqrt(K.abs(tensor)+1e-9))(prduct_fn1) #squre_root of sqrt_fn1
sqrt_vgg2 = Lambda(lambda tensor  : K.sign(tensor)*K.sqrt(K.abs(tensor)+1e-9))(prduct_vgg1) #squre_root of sqrt_vgg1

    
added_vgg = Conv2D(128 , [1,1] )(added_vgg)
subtract_vgg = Conv2D(128 , [1,1] )(subtract_vgg)
subtract_vgg2 = Conv2D(128 , [1,1] )(subtract_vgg2)
prduct_vgg1 = Conv2D(128 , [1,1] )(prduct_vgg1)
sqrt_vgg1 = Conv2D(128 , [1,1] )(sqrt_vgg1)
sqrt_vgg2 = Conv2D(128 , [1,1] )(sqrt_vgg2)
    
#finally concatenating all the above featues for final layer which is to be inputed to the dense layers.
concatenated= Concatenate(axis=-1)([Flatten()(added_vgg), (added_facenet), Flatten()(subtract_vgg), (subtract_fn),
                                   Flatten()(subtract_vgg2), (subtract_fn2), Flatten()(prduct_vgg1), (prduct_fn1), 
                                   Flatten()(sqrt_vgg1), (sqrt_fn1), Flatten()(sqrt_vgg2), (sqrt_fn2)])
    
concatenated= Dense(500, activation="relu")(concatenated)
concatenated= Dropout(0.1)(concatenated)
concatenated= Dense(100, activation="relu")(concatenated)
concatenated= Dropout(0.1)(concatenated)
concatenated= Dense(25, activation="relu")(concatenated)
concatenated= Dropout(0.1)(concatenated)
out = Dense(1, activation="sigmoid")(concatenated) #output sigmoid layer

#defining the model
model = Model([input_1, input_2, input_3, input_4], out)



## =============


#this function will read image from specified path and convert it into required size
def read_img_fn(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(IMG_SIZE_FN,IMG_SIZE_FN))
    img = np.array(img).astype(np.float)
    return prewhiten(img)

#this function will read image from specified path and convert it into required size
def read_img_vgg(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(IMG_SIZE_VGG,IMG_SIZE_VGG))
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)



#generator funtion will generate images in the right format while training the model 
def generate(list_tuples, person_to_images_map, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 2)
        labels = [1] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            p1 = choice(ppl)
            p2 = choice(ppl)

            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        X1_FN = np.array([read_img_fn(x) for x in X1])
        X1_VGG = np.array([read_img_vgg(x) for x in X1])

        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2_FN = np.array([read_img_fn(x) for x in X2])
        X2_VGG = np.array([read_img_vgg(x) for x in X2])

        yield [X1_FN, X2_FN, X1_VGG, X2_VGG], labels    
