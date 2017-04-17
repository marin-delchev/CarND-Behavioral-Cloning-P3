
**Behavioral Cloning Project**
---
[//]: # (Image References)

[image1]: ./examples/data.png "Data distribution"
[image2]: ./examples/image.png "Example image"

The project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

The car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
And running the simulator on autonomous mode on the first track.

The model uses the following architecture: four convolution layers, followed by max pooling, flatten layer and
two dense layers. To prevent over-fitting two dropout layers are used. Sizes of the convolutions, strides and activations used
are described in the following snippet:

```
model.add(Cropping2D(cropping=((60, 10), (0, 0))))
model.add(Convolution2D(16, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'))
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'))
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'))
model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid', activation='elu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
```

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                    
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]            
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]                  
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 16)   1216        cropping2d_1[0][0]              
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 77, 32)    12832       convolution2d_1[0][0]           
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 37, 64)     51264       convolution2d_2[0][0]           
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 18, 64)     36928       convolution2d_3[0][0]           
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 1, 9, 64)      0           convolution2d_4[0][0]           
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 576)           0           maxpooling2d_1[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 576)           0           flatten_1[0][0]                 
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 256)           147712      dropout_1[0][0]                 
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 256)           0           dense_1[0][0]                   
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 10)            2570        dropout_2[0][0]                 
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             11          dense_2[0][0]                   
====================================================================================================
Total params: 252,533
Trainable params: 252,533
Non-trainable params: 0
____________________________________________________________________________________________________
```

The model uses Adam optimizer with learning rate of 0.001. Other learning rates were tested but the
results were not as good.

```
adam = Adam(lr=0.001)
model.compile(loss='mse', optimizer=adam)
```

Training data:

Different approaches were tested for collecting the dataset.
```
1 - Recording 10 laps on the first track.
2 - Recording 2 reverse laps on the first track.
3 - Recording position recovery.
```

However, the provided sample dataset was superior to the collected data.
It consists of 8036 images.
An example image from the central camera.

![alt text][image2]

The angle distribution of the provided sample dataset is the following.


![alt text][image1]

The most important part of the task was the data augmentation.
First, each image is flipped and the angle is multiplied by -1.
The images from the sides cameras are used with the angle correction of 0.2.
The angle correction is added to the angle for the left image and subtracted from the angle for the
right image. This is an effective way of preventing the car from leaving the track.
In addition, top 60 pixels and bottom 10 pixels are cropped because they are mostly noise.


Solution approach

The first model architectures that were tested were LeNNet and Nvidia`s network.
However, both architectures were failing to control the car after the bridge on the first track.
I tried to collect more data at the failing point, however, that did not improve the results.

Then I attempted to convert the images to other color schemes (Gray, HSV, etc.). That did not work as well.

Using many various network architectures and 90/10% split into training and test datasets I arrived at the architecture
described above. For training I used AWS GPU instance.
