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