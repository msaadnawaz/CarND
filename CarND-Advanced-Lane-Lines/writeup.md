## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chessboard_corners.jpg "Chessboard Corners"
[image2]: ./output_images/chessboard_undistorted.jpg "Chessboard Undistorted"
[image3]: ./output_images/test_image.jpg "Test Image"
[image4]: ./output_images/combined_gradient.jpg "Combined Gradient"
[image5]: ./output_images/straight_warp.jpg "Straight Line Warp"
[image6]: ./output_images/polyfit.jpg "Polynomial Fit"
[image7]: ./output_images/overlay.jpg "Overlayed Image with Highlighted Lane"
[image8]: ./output_images/shadow_overlay.jpg "Overlayed Image for Shadowed Road"
[image9]: ./output_images/clutter.jpg "Clutter on Road"
[video1]: ./project_video_ouput.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code cells 2 to 5 of the IPython notebook "Image_pipeline.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I overlayed the detected corners back on to chessboard image. Example shown:

![alt text][image1]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test chessboard image using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at code cells 10 through 16 in `Image_pipeline.ipynb`).  First I only thresholded saturation channel of HLS color space and it showed good results on images but it as not showing good results on shadowed parts of lanes in video. Then I thresholded the lighting channel as ell and it showed good results for those corner cases as well. Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `pers_trans()`, which appears in code cell 17 in the same file.  The `pers_trans()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
img_size = (undistorted_images[0].shape[1], undistorted_images[0].shape[0])

#src is the portion of road where we want to find lane lines
src = np.float32([[np.int(img_size[0]*0.435), np.int(img_size[1]*0.65)],
                  [np.int(img_size[0]*0.565), np.int(img_size[1]*0.65)],
                  [np.int(img_size[0]*0.9), img_size[1]],
                  [np.int(img_size[0]*0.1), img_size[1]]])
offset = 150

#spread road area on complete destination perspective transformed image
dst = np.float32([[offset, 0], 
                  [img_size[0]-offset, 0], 
                  [img_size[0]-offset, img_size[1]], 
                  [offset, img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 556, 468      | 150, 0        | 
| 723, 468      | 1130, 0       |
| 1152, 720     | 1130, 720     |
| 128, 720      | 150, 720      |

I verified that my perspective transform was working as expected by drawing the warped counterpart of straight lines image and verified that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used max values of histogram across x-axis to find the lane lines and used these points on both sides of center of images to fit the polynomial on lane lines. Then I collected x and y indices of non-zero values and divided y-axis into 9 search windows. I defined 100 pixels as margin on either sides of center (histogram max index) to create first window and collected all the so-called "good indices which occured within the window. Then I recalculated center for next window by taking mean of all the good x indices of current windo. Similarly I repeated this exercise for all 9 windows and concatenated the good indices. Then I collected all the x and y values for good indices of left and right lane lines to fit second order polynomials for both the lane lines like this:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in code cell 24 in same file. As camera is assumed to be mounted in center of car so mid of image is also the mid of car and with this assumption I calculated the distance of right and left lane lines in pixels. Then I calculated absolute difference of the two distances and then converted this value into meters to get the position of car from center of lane.

For radius of curvature, I first calculated the polynomials again in world space by converting the pixels into meters and then used the given formula to calculate the radius of curvature.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in code cell 25 in `Image_pipelne.ipynb`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I combined the gradient and HLS thresholds to extract lane lines from images and then I warped the images such that the straight lines appear parallel and vertical on warped image. Similarly I went ahead with procudure as mentioned but I tried this on over all images successfully. But I still didn't get to run smoothly on the video and then I changed my pipeline for video to average the result of 10 recent frames. Then I changed my video pipeline to start searching for lane lines in next frame from the base (y=720) of x value in current frame but this method made the lane finding worse because I was not considering the histogram output of current frame and there was significant difference in base x value for some frames. After this, I changed the pipeline to start searching for lane line from the mean of previous frame base value and histogram max value. This helped my lane finding significantly but I was still going off in shadows. Then I started thresholding over lightness channel and tried it on captured screenshot from video as well which helped me run over this smoothly.

[alt text][image8]

My implementation does not work good when there are more markings on road than the actual lane lines (like in challenge video). It partially works when there are hard turns like in harder challenge video.

The pipeline could be made more robust if the polynomial length over y-axis is kept variable to accomodate sharp turns. One suggestion from lesson which I could not successfully implement is to check whether the lane lines are parallel or not. I believe this would also make it robust. Another problem is removal of clutter i.e. consistent shadows from road dividers/guardrails and renovated patches of road which appear as additional line.

[alt text][image9]