#**Finding Lane Lines on the Road** 

##Writeup Submission by Muhammad Saad Nawaz

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./output_images/edges.jpg "Edges"
[image2]: ./output_images/masked.jpg "Masked"
[image3]: ./output_images/separate_lines.jpg "Separate Lines"
[image4]: ./output_images/separate_output.jpg "Separate Lines Output"
[image5]: ./output_images/lines.jpg "Lines"
[image6]: ./output_images/output.jpg "Output"
[image7]: ./output_images/roi.jpg "Region of interest"
[image8]: ./output_images/output2.jpg "Output"
[image9]: ./output_images/output3.jpg "Output"
[image10]: ./output_images/output4.jpg "Output"
[image11]: ./output_images/output5.jpg "Output"
[image12]: ./output_images/output6.jpg "Output"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I blurred(smoothened) the image using Gaussian Blur. Afterwards, I applied Canny Edge Detection to extract the sharp edges:
![alt text][image1]

Then, I defined a polygon as region of interest where the lane lines are expected to appear. This is lower 40% of image and complete horizontal axis inside region of interest and reducing to only 10% (between 45% and 55% of image) going to the top of region of interest: 
![alt text][image7]

After this I masked the image with this region of interest.
![alt text][image2]

Then, I used Hough transform to extract lines from the masked edges and got the separate lines for each lane marking.
![alt text][image3]

At last I overlayed the lane markings on the actual image to get this:
![alt text][image4]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by separating all the lines on left lane lines from the lines on right lane lines. Then I calculated slopes for all the lines on each side and then averaged the slopes for each side. Then, I calculated intercepts for each of the two lines using average slope for each line and one point on line. At last, I projected the line to the bottom of image on one side and to the top of region of interest on the other side. Then, I used these projected points to draw lines.

To optimize the lane line finding on video, I created a list of 10 recent frames and averaged the output line across these 10 frames.

Here is the final output:
![alt text][image6]

Here is the output of all other test images:
![alt text][image8]

![alt text][image9]

![alt text][image10]

![alt text][image11]

![alt text][image12]


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when there is a sharp curve. Another shortcoming could be if there is a steep slope instead of a plain road. Currently my algorithm doesn't give output for the reflections of bumper (challenge video) and expects whole frame to be part of road.


###3. Suggest possible improvements to your pipeline

A possible improvement would be to use a curve/arc as output instead of a straight line to cater curved/steep roads.

Another potential improvement could be to filter out the bumper if it is visible in camera video.