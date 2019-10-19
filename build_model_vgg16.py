import os
import cv2
import scipy
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping



def get_data(folder_PATH):
    x=[]
    y=[]
    for i in os.listdir(folder_PATH):
        if i[0]=='c':
            label=0
        elif i[0]=='d':
            label=1
        else:
            print("u r giving wrong image...................................")

        image_PATH=folder_PATH+i
        image_FILE=cv2.imread(image_PATH)
        image_FILE=scipy.misc.imresize(arr=image_FILE, size=(227, 227, 3))
        image_FILE=np.asarray(image_FILE)

        x.append(image_FILE)
        y.append(label)
    x=np.asarray(x)
    y=np.asarray(y)

    return x,y


train_folder_PATH="/home/djkhai/Desktop/DataSet/dog-vs-cat/train/"
val_train_PATH="/home/djkhai/Desktop/DataSet/dog-vs-cat/val/"

train_x, train_y=get_data(train_folder_PATH)
val_x, val_y= get_data(val_train_PATH)


# Encoding labels to hot vectors
train_y_hot = to_categorical(train_y, num_classes = 2)
val_y_hot = to_categorical(val_y, num_classes = 2)



model = VGG16(weights='/home/djkhai/Desktop/keras_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(227,227,3))
#final_model_PATH="/home/djkhai/PycharmProjects/Kaggle/dog_vs_cat/model_file/dog-cat.h5"

for layer in model.layers[:15]:
    layer.trainable = False


#Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(4096, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model
model_final = Model(input = model.input, output = predictions)

# compile the model
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])


bat_size = 200
max_epochs = 1  # too few
print("Starting training ")
model_final.fit(train_x, train_y_hot, batch_size=bat_size,epochs=max_epochs, verbose=1)
print("Training complete")

model_final.save('dog-cat.h5')  # creates a HDF5 file 'my_model.h5'

# 4. evaluate model
loss_acc = model_final.evaluate(val_x, val_y_hot, verbose=0)
print("\nTest data loss = %0.4f  accuracy = %0.2f%%" % \
  (loss_acc[0], loss_acc[1]*100) )
