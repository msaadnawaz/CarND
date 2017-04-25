**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Shape"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/normal.jpg "Normal Image"
[image4]: ./examples/recovery1.png "Recovery Image"
[image5]: ./examples/recovery2.png "Recovery Image"
[image6]: ./examples/recovery3.png "Recovery Image"
[image7]: ./examples/recovery4.png "Recovery Image"
[image8]: ./examples/recovery5.png "Recovery Image"
[image9]: ./examples/recovery6.png "Recovery Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network on EC2 instance and model-pc.py file contains code for training and saving the convolution neural network on Windows PC. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. There are only two differences between the two code files; one is the way of accessing the training images because of different locations of training data for both machines; other difference is that, in EC2 machine, cropping is done on images before creating numpy arrays from them to save memory while, in PC, cropping is done using Keras Cropping layer.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution layers with 3x3 filter sizes and depths between 18 and 96 (model.py lines 120-127). I use Maxpool layers after each convolution layer to reduce dimensionality.

The model includes RELU activations within convolution layers to introduce nonlinearity, the data is normalized in the model using a Keras lambda layer (code line 119) and cropped using Kera Cropping layer (code line 110- only in model-pc.py). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 129). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 106). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 143).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road as well as wrong direction driving.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to increase the depth with each convolution layer to learn more features through more filters.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it had two convolution layers to learn good amount of features.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model and added two more convolution layers and two more fully-connected layers

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I added more training data for these turns.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 118-140) consisted of a convolution neural network with the following layers and layer sizes:

![alt text][image1]

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to. These images show what a recovery looks like starting from road edge:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]


To augment the data sat, I also flipped images and angles thinking that this would minimize bias.

After the collection process, I had 52278 number of data points. I then preprocessed this data by normalizing and croppping the images

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by low MSE. I used an adam optimizer so that manually training the learning rate wasn't necessary.
