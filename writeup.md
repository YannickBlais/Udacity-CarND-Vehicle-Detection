**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Normalize features and randomize a selection for training and testing.
* Sliding-window implementation technique and use trained classifier to search for vehicles in images.
* Pipeline run on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car-not-car.png
[image2]: ./examples/Car_features.png
[image3]: ./examples/Not-Car_features.png
[image4]: ./examples/YCrCb_HOG.png
[image5]: ./examples/32x32.png
[image6]: ./examples/48x48.png
[image7]: ./examples/64x64.png
[image8]: ./examples/128x128.png
[image9]: ./examples/all_windows.png
[image10]: ./examples/all_windows.png
[image11]: ./examples/heat_map.png
[image12]: ./examples/labels.png
[image13]: ./examples/final_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function get_hog_features() lines XXX through XXX of the file called `lesson_function.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]
![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled on parameters that seemed to work best for the given video: 
- orient = 32
- pixels per cell = 16
- cells per block = 2
- color space = 'YCrCb', using all channels
I chose more orientations, but also more pixels per cell (compared with the course) which is is quite fast but still has enough information to obtain good results. Here is an example of HOG image with those parameters:

![alt text][image4]


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM sklear's LinearSVC package. This can be found in file training.py, function train_model(), lines XXX through XXX. Moreover, I used GridSearchCV to attempt to try to find the best parameters. I originally used the same parameters as described in sklearn's documentation (with svm.SVC) with 'linear' and 'rbf' kernels, but I found it was quite slow, so I set my parameters simply as this:
```
parameters = {'C': [1, 10]}
```
However the value chosen by the algorithm was '1', the default value!

Second, I also use 32x32 spatial features, which is done by simply resizing the image to 32x32 pixels. This task is performed in file lesson_functions.py, lines XXX through XXX.

Third, I used the histogram features with all 3 channels, 32 bins and the whole 8-bits (256 levels) range and this can be found in lesson_functions.py lines XXX through XXX.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window sizes of 32x32, 48x48, 64x64, and 128x128. This code is in file detect_vehicles.py, lines XXX through XXX. The different window sizes are represented in this image:

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

I chose those boxe sizes to perform searches at many different scales to maximize the chance of finding a match, and since the classification was rather fast compared with other options that I tried, I could use a lot of them. The smaller boxes search more on locations in the road far away while the bigger ones on locations nearby the driver. I mostly concentrated the search in the center where I often lost track of the white car (see discussion for more on this).

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As explained previously, I searched on 4 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image9]
![alt text][image10]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4).


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As proposed in the course, I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

On top of this, as an attempt to filter out false classifications that appear on a single frame, if a box in the current frame does not match a box found in the previous frame, i.e. the center of current box is not somewhere inside a box on previous frame, it is removed.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image11]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image12]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image13]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

