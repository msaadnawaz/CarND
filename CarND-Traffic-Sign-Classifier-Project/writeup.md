#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/image2.jpg "Visualization"
[image2]: ./output_images/example.jpg "images"
[image4]: ./traffic-sign-test-images/image1.jpg "Traffic Sign 1"
[image5]: ./traffic-sign-test-images/image2.jpg "Traffic Sign 2"
[image6]: ./traffic-sign-test-images/image3.jpg "Traffic Sign 3"
[image7]: ./traffic-sign-test-images/image6.jpg "Traffic Sign 4"
[image8]: ./traffic-sign-test-images/gsol.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/msaadnawaz/CarND/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the built-in python functions and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is split across different classes.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale to train the model without influence of color sharpness.


As a last step, I normalized the image data to bring the values to the range of -1 to 1 and prevent model from going out of bounds from large values .

Here is an example of a traffic sign image before and after preprocessing.

![alt text][image2]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The training and validation sets were already split from the dataset and the code for reading the data is contained in the first code cell of the IPython notebook.  

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the tenth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image 						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten       		| outputs 400  									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|
| Softmax				|           									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the fourteenth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the fifteenth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
    LeNet architecture was chosen
* Why did you believe it would be relevant to the traffic sign application?
    I learned and implemented LeNet architecture during CNN lesson and MNIST lab and I wanted to learn to tune the parameters and       hyperparameters connected around the architecture before changing the architecture itself. The LeNet architecture with two convolutional layers and 3 fully connected layers should be fully capable of splitting images into 43 classes.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    The final model gives X percent accuracy for training data and Y percent accuracy for validation data which means that it does pretty well in classification of traffic sign images.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 
![alt text][image5] 
![alt text][image6] 
![alt text][image7] 
![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep Right  			| Keep Right									|
| Stop Sign      		| Stop sign   									| 
| Right of Way			| Right-of-way at the next intersection			|
| 50 km/h	      		| 50 km/h   					 				|
| Go Straight or Left	| Go Straight or Left  							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a keep right sign (probability of ~1.0), and the image does contain a keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1.00000000  			| Keep right   									| 
| 5.17256593e-09		| No passing for vehicles over 3.5 metric tons	|
| 7.53621998e-10		| Turn left ahead								|
| 3.98585550e-14		| Yield											|
| 2.37154137e-16		| Speed limit (80km/h)							|


For the second image, the model is pretty sure that this is a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1.00000000  			| Stop sign   									| 
| 5.75215639e-11		| Speed limit (80km/h)							|
| 1.70284620e-11		| Speed limit (50km/h)							|
| 3.59226300e-14		| Speed limit (30km/h)			 				|
| 2.20234792e-15		| Yield											|


For the third image, the model is pretty sure that this is a Right-of-way at the next intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1.00000000  			| Right-of-way at the next intersection			| 
| 2.35167892e-22		| Beware of ice/snow							|
| 9.44370732e-25		| Traffic signals								|
| 4.85561028e-31		| Double curve					 				|
| 2.50079248e-36		| Turn right ahead								|


For the fourth image, the model is pretty sure that this is a Speed limit (50km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1.00000000  			| Speed limit (50km/h)							| 
| 1.91813271e-22		| Wild animals crossing							|
| 7.31529271e-23		| Speed limit (80km/h)							|
| 3.70683505e-27		| Speed limit (30km/h)			 				|
| 7.91947569e-29		| Bicycles crossing 							|


For the fifth image, the model is pretty sure that this is a Go straight or left sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1.00000000  			| Go straight or left							| 
| 3.51341406e-22		| Slippery road									|
| 2.84842709e-25		| General caution								|
| 2.79677311e-28		| Ahead only					 				|
| 1.68047724e-29		| Dangerous curve to the right					|