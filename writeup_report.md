#Behavioral Cloning

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

Behavioral Cloning Project

The goals / steps of this project are the following:

Use the simulator to collect data of good driving behavior
Augment the data set to simulate various conditions
Build, a convolution neural network in Keras that predicts steering angles from images
Train and validate the model with a training and validation set
Test that the model successfully drives around track one without leaving the road
Summarize the results with a written report
Rubric Points

###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

model.py containing the script to create and train the model
drive.py for driving the car in autonomous mode
model.h5 containing a trained convolution neural network
writeup_report.md or writeup_report.pdf summarizing the results
video.mp4 recorded video of autonomous driving on track 1
video_track2.mp4 recorded video of autonomous driving on track 2

####2. Submission includes functional code Using the Udacity provided simulator and my model.py file for training the model. 
Using drive.py the car can be driven autonomously around the track by executing

python drive.py model.h5
####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I have used Nvidia end-to-end network given in the paper
The model uses 5 convolutional layer followed by 4 fully connected layers.
Activation function of ReLu is used between each Neural Network layer.
Before Convolutional layers, I have used two layers using Keras:
1. layer of normalization
2. layer for cropping the images to extract the region of interest

The figures shows the layers of the model.

####2. Attempts to reduce overfitting in the model
The data set is spit as follows:
training dataset = 0.8 
validation dataset = 0.2
I am using the splitting feature of model.fit_generator.
I am plotting the training and validation loss using the history feature of the model.fit_generator.
Initial training, resulted in the validation loss increasing when the training loss was continuously decreasing.
This is a clear case of overfitting. The network is not able to generalize.
The following graph depicts it:
Therefore I used dropout layers in the fully connected layers with a dropout probability of 0.5.


The model was trained and validated on different data sets to ensure that the model was not overfitting. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the following approach broadly:
1. Data augmentation of existing data set
2. Data collection by driving in the opposite direction for Track 1
3. Data collection by driving in right lane on Track 2 in both normal and opposite direction

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The strategy was to generate enough dataset which simulate the scenarios that the car would encounter on the track.
1. Right turns
2. Left turns
3. Recovering back to center if getting close to the lanes
4. Different lighting conditions

I used Nvidia End-to-End neural network as it is proven to work on On-Road tests.

First I tried training the network with the data of track1 provided in the project using a model generator 
The generator function was initially  simply reads images and uses them as they were for training.
The model includes normalization and cropping layers before convolution layers. 
The splitting feature of the model.fit_generator was used to split the data set.
The plotting of training and validation loss was done using the history feature of the model.fit_generator.
It was observed that the validation loss was increasing when the training loss was continuously decreasing.
This is a case of overfitting. The model is not able to generalize.
I employed the dropout layers for fully connected layers.
The validation loss improved after that.
The following graphs show the improvement
When I tried the model on the simulator in autonomous mode, it was not turning for the second left turn and it was
going into the muddy road.

####2. Creation of the Training Set & Training Process

I plotted a histogram of the steering angles. The plot shows that the instances of steering angles is very high as compared to other.
The model had not learned for going straight as there was less data set corresponding to taking turns.

To increase the dataset I employed data augmentation.
For taking turns in the track 1, the approximate steering angle required was around +/-0.2 for which the dataset was less.
For data augmentation I used generator function feature of model.fit_generator which would generate the augmented dataset
on the fly. If we generate the augmented images separately and store, the image array size would be huge and python would run out of memory.

1)I used the left and right camera images and simulated the instances of taking left and right turn by adding a correct factor of 0.25 in the steering angle.
2)In a single iteration I randomly selected one image -center, left or right. This would generate a data set with even distribution of various stering angle.
3)To even out the data set for left and right turns, I randomly flipped an image and multiplied (-1) to the corresponding steering angle.
4)To simulate different lighting conditions, I changed the brightness of the images by randonly choosing a factor.

With the model was doing better with the turns. But it was turning very closely for the last right turn in the track 1.
It was observed that the track 1 has 3 left turns and 1 right turn.
It seemed that even with randomly picking left,right or images and randomly flipping was not generating enough dataset
for right turns and the base dataset itself is biased towards left turns.
To overcome this by increasing dataset using simulator in training mode by driving in the opposite directions.
The opposite track gave 3 right turns and 1 left turn.
After this experimentation was working well for track 1.

I tried the model on track 2. It did not give very good results.
1)The track 2 is more difficult than track 1 as it has many turns which far curvier than the track 1.
2)There are many instances where the orientation of the road is not straight w.r.t the image because of curcy inclines and declines.
3)Track 2 has a divider lane
4)There are many patches of road where there are very dark shadows.
The track 1 training data did not train the model for such scenarios.

I collected more data by driving the simulator on track 2.
Adding data set of track 2 to the base data set helped the model to learn the difficult features associated with track 2.
It covered the track 2 to a good extent.

Another point that I observed that with adding the track 2 data set, the driving on track 1 had also improved for the turns.
Earlier where the car would close to the lanes while taking turns. In this case, the cars were quite centered even while taking turns.
This should be because it is trained with more data set of left and right turns.

####3. Reflections
Many more techniques for data augmentation to the base data set can be employed like rotation, translation, artificial shadowing, etc.
This would reduce the amount of manual driving that is needed for collecting data set.
