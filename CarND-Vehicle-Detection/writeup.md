**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./output_images/hog1.png
[image3]: ./output_images/features_before_normalization.png
[image4]: ./output_images/normalized_features.png
[image8]: ./examples/sliding_window.jpg
[image9]: ./examples/bboxes_and_heat.png
[image10]: ./examples/labels_map.png
[image11]: ./examples/output_bboxes.png
[video1]: ./project_video_output.mp4


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the step one of the IPython notebook trianing_pipeline.ipynb (code cell 8).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. The original image and hog extraction of all three channels is shown:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that YUV and YCrCb have better performance on images and where I can use all three color channels. I tried 9 and 12 orientations and settled with 12.

I verified that normalization is working well on features and this helped in deciding for including histogram and spatial binned features as well.

![alt text][image3]
![alt text][image4]

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I splitted the data into training and testing sets and then I trained a linear SVM using normalized features. The code is in step two of training _pipeline.ipynb.

I checked the test accuracy and it was 99.21%

### Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented this in Detectionpipeline.ipynb of my repository. I marked region of interest from 400th to 656th pixel on y-axis and all pixels on x-axis and then extracted all the hog features for this region. Then I started searching for each window from left to right and top to bottom for the hog, histogram and spatial features. I then fed this into my classifier to find out the result of prediction on the window. If positive, I appended it into the lise of hotwindows to be used for drawing later.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image8]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Later I also used the centroids of previous detections to add heat to the region where the detection was probable. I added this to add weight in favor of detection and create further difference between correct detections and false positives.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image9]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image10]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image11]

---

### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The first problem I found was that my findcars function had different feature extraction than in training so the data was not getting correctly normalized hence showing huge number of false positives. Then, I had to do several iterations to find a color space with minimum false positives. With correction of findcars function and settling with YCrCb, I moved on to generate the complete pipeline with video output.

Even after setting a threshold of false positives, I was still seeing a lot of them. Then, I started caching 10 recent boxes output. And then I added them to the heatmap and set a threshold on all the 10 recent and current boxes. This helped me in removing a lot of false positives but I was still getting some strong detections from bridge guardrails. Then I added the recent correctly detected labels as well to the heatmap to further reduce the false positives but this caused to lose the car when it is far from us.

Then I added 3 more scales to make total four sliding window searches. First on 32x32 windows only between pixels 400 to 528 on y-axis, second on 48x48 windows between pixels 400 to 560, third on 64x64 windows between pixels 400 to 592 and fourth on 96x96 windows between pixels 400 and 656. The idea for this was to be distinctly able to detect distant cars but also not add to false positives by searching small windows throughout the space.

In the meanwhile, I tried several other ways to detect cars without false positives which didn't work (most of them).

