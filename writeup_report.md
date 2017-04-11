#**Behavioral Cloning Project**

[//]: # (Image References)

[image1]: ./images/no_dropouts.png "Loss without Dropouts"
[image2]: ./images/dropouts.png "Loss after Dropouts"
[image3]: ./images/hist.png "Histogram of baseline dataset"
[image4]: ./images/hist_added_data.png "Histogram with added dataset"
[image5]: ./images/center.jpg "Center Image"
[image6]: ./images/left.jpg "Left Image"
[image7]: ./images/right.jpg "Right Image"
[image8]: ./images/flip.jpg "Flipped Image"
[image9]: ./images/brightness.jpg "Brightness adjusted Image"
[image10]: ./images/model.png "Nvidia Model"
[image11]: ./images/loss.png "Loss"

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

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results
* video_track1.mp4 recorded video of autonomous driving on track 1
* video_track2.mp4 recorded video of autonomous driving on track 2
* run_track1.mp4 center camera recorded video of autonomous on track 1
* run_track1.mp4 center camera recorded video of autonomous on track 2
* driving_log.csv csv file for dataset used

####2. Submission includes functional code Using the Udacity provided simulator and my model.py file for training the model. 
Using drive.py the car can be driven autonomously around the track by executing

python drive.py model.h5

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I have used Nvidia end-to-end network given in the paper (line no. 103-113):
https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

The model uses 5 convolutional layer followed by 3 fully connected layers.
The last fully connected layer is connected to a single neoron for predicting the steering angle.
Activation function of ReLu is used between each Neural Network layer.
Before Convolutional layers, I have used two layers using Keras:
1. layer of normalization
2. layer for cropping the images to extract the region of interest

####2. Attempts to reduce overfitting in the model
The data set is spit as follows:
training dataset = 0.8 
validation dataset = 0.2
I am using the splitting feature of model.fit_generator.(line no. 128)
I am plotting the training and validation loss using the history feature of the model.fit_generator.
Initial training, resulted in the validation loss increasing when the training loss was continuously decreasing.
This is a case of overfitting. The network is not able to generalize.

Therefore I used dropout layers in the fully connected layers with a dropout probability of 0.5. (line no. 112-117)
The model was trained and validated on different data sets to ensure that the model was not overfitting. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.(line no. 124)

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the following approach broadly:
1. Data augmentation of existing bsaeline data set
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
https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

![alt text][image10]

| Layer         		    |     Description	        					                       | 
|:---------------------:|:--------------------------------------------------------:| 
| Input         		    | 160x320x3 RGB image   							                     |
| Normalization         |                                                          |
| Cropping              |                                                          |
| Convolution 5x5     	| 24 kernels  	                                           |
| RELU					 				|                                                          |
| Convolution 5x5      	| 36 kernels                                				       |
| RELU					 				|                                                          |
| Convolution 5x5      	| 48 kernels                                				       |
| RELU					 				|                                                          |
| Convolution 5x5      	| 64 kernels                                				       |
| RELU					 				|                                                          |
| Convolution 5x5      	| 64 kernels                                				       |
| RELU					 				|                                                          |
| Fully connected		100 |                   									                     |
| RELU					 				|                                                          |
| Fully connected		50  |                 									                       |
| RELU					 				|                                                          |
| Fully connected		10  |                 									                       |
| RELU  				        |             									                           |
| Fully connected		1   |                 									                       |

First I tried training the network with the data of track1 provided in the project using a model.fit_generator 
The generator function was initially  simply reading images and useing them as they were for training.
The model includes normalization and cropping layers before convolution layers.
RGB images are given as input to the network for training.
The splitting feature of the model.fit_generator was used to split the data set.
The plotting of training and validation loss was done using the history feature of the model.fit_generator (line no. 128-140).
It was observed that the validation loss was increasing when the training loss was continuously decreasing.
This is a case of overfitting. The model is not able to generalize.
I employed the dropout layers for fully connected layers.
The validation loss improved after that.
The following graphs show the improvement:

Without Dropouts:

![alt text][image1] 

After applying dropouts:

![alt text][image2]

When I tried the model on the simulator in autonomous mode, it was not turning for the second left turn and it was
going into the muddy road.

####2. Creation of the Training Set & Training Process

I plotted a histogram of the steering angles. The plot shows that the instances of 0 steering angles is very high as compared to other.
The model had mostly learned for going straight and had not learned for taking turns as there was less data set corresponding to taking turns.

![alt text][image3]


To increase the dataset I employed data augmentation. I should thank this [blog post](https://medium.com/@subodh.malgonde/teaching-a-car-to-mimic-your-driving-behaviour-c1f0ae543686) for suggestions about data augmentation.
For taking turns in the track 1, the approximate steering angle required was around +/-0.2 for which the dataset was less.
For data augmentation I used generator function feature of model.fit_generator which would generate the augmented dataset
on the fly. If we generate the augmented images separately and store, the image array size would be huge and python would run out of memory.

1)I used the left and right camera images and simulated the instances of taking right and left turn respectively by adding a correction factor of 0.25 to the steering angle.
2)In a single iteration I randomly selected one image -center, left or right. This would generate a data set with even distribution of various steering angles.
3)To even out the data set for left and right turns, I randomly flipped an image and multiplied (-1) to the corresponding steering angle.
4)To simulate different lighting conditions, I changed the brightness of the images by randomly choosing a brightness multiplication factor.

The following are center, left and right images:

![alt text][image5] ![alt text][image6] ![alt text][image7]

The following show flipping of images:

![alt text][image5] ![alt text][image8]

The following show brightness adjustment of images:

![alt text][image5] ![alt text][image9]

With this experiment, model was doing better with the turns. But it was turning very closely for the last right turn in the track 1.
It was observed that the track 1 has 3 left turns and 1 right turn.
It seemed that even with randomly picking left,right or images and randomly flipping was not generating enough dataset
for right turns and the base dataset itself is biased towards left turns.
To overcome this by increasing dataset using simulator in training mode by driving in the opposite directions.
The opposite track gave 3 right turns and 1 left turn.
After this experimentation the model was working well for track 1 for right turn as well.
The basic goal is to train the model for recovering to the center.

I tried the model on track 2. It did not give very good results.
1)The track 2 is more difficult than track 1 as it has many turns which are far curvier than the track 1.
2)There are many instances where the orientation of the road is not straight w.r.t the image because of curcy inclines and declines.
3)Track 2 has a divider lane
4)There are many patches of road where there are very dark shadows.
The track 1 training data did not train the model for such scenarios.

I collected more data by driving the simulator on track 2. The histogram plot for the new data set:

![alt text][image4]

The data set includes a lot more left and right turns.
Following is the plot for loss:

![alt text][image11]

The total training data set count is 83757 including left,right and center images.
Considering this as the total count I used 83757x0.8 ~ 67000 for training and 83757x0.2 ~ 16751.

* Epochs 5
* Training set 67000
* Validation set 16751
* Optimizer Adam

Although the augmented data set would contain flipped images and brightness adjusted images too.
Adding data set of track 2 to the base data set and augmenting on the fly helped the model to learn the difficult features associated with track 2. It covered the track 2 to a good extent but crashed later

Another point that I observed that with adding the track 2 data set, the driving on track 1 had also improved for the turns.
Earlier where the car would close to the lanes while taking turns. In this case, the car was quite centered even while taking turns.
This should be because it is trained with more data set of left and right turns and recovery data.

####3. Reflections
* Many more techniques for data augmentation to the base data set can be employed like rotation, translation, artificial shadowing, etc.
* More data set can also be generated for track 2 by recording the recovery from left to right lane.
  I had collected data of driving only on the right lane of track 2.
* Keras simplifies coding. With very few lines of code one can model a complex neural network architecture
