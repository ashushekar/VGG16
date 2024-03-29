# Step by step VGG16 implementation in Keras

VGG16 is a convolution neural net (CNN ) architecture which was used to win ILSVR(Imagenet) competition in 2014. 
It is considered to be one of the excellent vision model architecture till date. Most unique thing about VGG16 
is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 
filter with a stride 1 and always used same padding and maxpool layer of 2x2 filter of stride 2. 
It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. 
In the end it has 2 FC(fully connected layers) followed by a softmax for output. The 16 in VGG16 refers to it has 
16 layers that have weights. This network is a pretty large network and it has about 138 million (approx) parameters.

Very Deep Convolutional Networks for Large-Scale Image Recognition. [https://arxiv.org/abs/1409.1556]

![vgg16 architecture](https://user-images.githubusercontent.com/35737777/69682136-5bdd4780-10a8-11ea-9079-50283f5451df.png)

The implementation of VGG16 can be done on Cats vs Dogs dataset.

### Packages Needed

```python
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.backend.tensorflow_backend import set_session
```
We will be using Sequential method which means that all the layers of the model will be arranged in sequence. Here we 
have imported ImageDataGenerator from _keras.preprocessing_. The objective of ImageDataGenerator is to import data with 
labels easily into the model. It is a very useful class as it has many function to rescale, rotate, zoom, flip etc. The 
most useful thing about this class is that it does not affect the data stored on the disk. This class alters the data on 
the go while passing it to the model.

### Image Data Generator

Let us create an object of _ImageDataGenerator_ for both training and testing data and passing the folder which has train
data to the object _trdata_ and similarly passing folder which has test data to the object of _tsdata_.

```python
trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="../Datasets/Cats&Dogs/train",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="../Datasets/Cats&Dogs/validation", target_size=(224,224))
```

The ImageDataGenerator will automatically label all the data inside cat folder as cat and vis-à-vis for dog folder. In 
this way data is easily ready to be passed to the neural network.

### Model Structure
```python
# Generate the model
model = Sequential()
# Layer 1: Convolutional
model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3),
                 padding='same', activation='relu'))
# Layer 2: Convolutional
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 3: MaxPooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 4: Convolutional
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 5: Convolutional
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 6: MaxPooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 7: Convolutional
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 8: Convolutional
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 9: Convolutional
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 10: MaxPooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 11: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 12: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 13: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 14: MaxPooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 15: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 16: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 17: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 18: MaxPooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 19: Flatten
model.add(Flatten())
# Layer 20: Fully Connected Layer
model.add(Dense(units=4096, activation='relu'))
# Layer 21: Fully Connected Layer
model.add(Dense(units=4096, activation='relu'))
# Layer 22: Softmax Layer
model.add(Dense(units=2, activation='softmax'))
```

Here we have started with initialising the model by specifying that the model is a sequential model. 
After initialising the model then we can add: 
1. 2 x convolution layer of 64 channel of 3x3 kernal and same padding
2. 1 x maxpool layer of 2x2 pool size and stride 2x2
3. 2 x convolution layer of 128 channel of 3x3 kernal and same padding
4. 1 x maxpool layer of 2x2 pool size and stride 2x2
5. 3 x convolution layer of 256 channel of 3x3 kernal and same padding
6. 1 x maxpool layer of 2x2 pool size and stride 2x2
7. 3 x convolution layer of 512 channel of 3x3 kernal and same padding
8. 1 x maxpool layer of 2x2 pool size and stride 2x2
9. 3 x convolution layer of 512 channel of 3x3 kernal and same padding
10. 1 x maxpool layer of 2x2 pool size and stride 2x2

We have also add ReLU activation to each layers so that all the negative values are not passed to the next layer.

After creating all the convolution we pass the data to the dense layer:

11. 1 x Dense layer of 4096 units
12. 1 x Dense layer of 4096 units
13. 1 x Dense Softmax layer of 2 units

#### Adam Optimizer
Let us use Adam optimiser to reach to the global minima while training out model. If we stuck in local minima while 
training then the adam optimiser will help us to get out of local minima and reach global minima. We will also 
specify the learning rate of the optimiser, here in this case it is set at 0.001. If our training is bouncing a lot on 
epochs then we need to decrease the learning rate so that we can reach global minima.

```python
# Add Optimizer
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
# Check model summary
print(model.summary())
```

#### Model Summary 

```sh 
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 224, 224, 64)      1792      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 224, 224, 64)      36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 112, 112, 64)      0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 112, 112, 128)     73856     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 112, 112, 128)     147584    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 56, 56, 128)       0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 56, 56, 256)       295168    
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 56, 56, 256)       590080    
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 56, 56, 256)       590080    
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 28, 28, 256)       0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 28, 28, 512)       1180160   
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 28, 28, 512)       2359808   
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 28, 28, 512)       2359808   
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 14, 14, 512)       0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 14, 14, 512)       2359808   
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 14, 14, 512)       2359808   
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 14, 14, 512)       2359808   
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 7, 7, 512)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 25088)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              102764544 
_________________________________________________________________
dense_2 (Dense)              (None, 4096)              16781312  
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 8194      
=================================================================
Total params: 134,268,738
Trainable params: 134,268,738
Non-trainable params: 0
_________________________________________________________________
```

### Model Implementation

#### Model Checkpoint Saving

ModelCheckpoint helps us to save the model by monitoring a specific parameter of the model. In this case we have monitoring 
validation accuracy by passing _val_acc_ to **ModelCheckpoint**. The model will only be saved to disk if the validation 
accuracy of the model in current epoch is greater than what it was in the last epoch.

```python
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', 
                             verbose=1, save_best_only=True, 
                             save_weights_only=False, mode='auto', period=1)
```

#### Early Stopping

EarlyStopping helps us to stop the training of the model early if there is no increase in the parameter which we have set 
to monitor in **EarlyStopping**. In this case we have monitoring validation accuracy by passing _val_acc_ to **EarlyStopping**. 
We have set patience to 20 which means that the model will stop to train if it does not see any rise in validation accuracy 
in 20 epochs.

```python
earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
```

#### Fit Generator
We are using _model.fit_generator_ as we have **ImageDataGenerator** to pass data to the model. We will pass train and test 
data to _fit_generator_. In _fit_generator_, _steps_per_epoch_ will set the batch size to pass training data to the model 
and validation_steps will do the same for test data. We can tweak it anytime based on our system specifications.

```python
hist = model.fit_generator(steps_per_epoch=100, generator=traindata, validation_data=testdata,
                           validation_steps=10, epochs=100,
                           callbacks=[checkpoint, earlystop])
```

### Plot Visualisation
We will visualise training/validation accuracy and loss using matplotlib.
![plot](https://user-images.githubusercontent.com/35737777/69684979-a6fc5800-10b2-11ea-874f-878037c95e74.png)

### Test the model

To do predictions on the trained model we need to load the best saved model and pre-process the image and pass the image 
to the model for output.

```python
img = image.load_img("../Datasets/Cats&Dogs/test1/39.jpg",target_size=(224,224))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)
saved_model = load_model("vgg16_1.h5")
output = saved_model.predict(img)
if output[0][0] > output[0][1]:
    print("cat")
else:
    print('dog')
```

![dog](https://user-images.githubusercontent.com/35737777/69684980-a6fc5800-10b2-11ea-896a-4f6f9c269e7b.png)
