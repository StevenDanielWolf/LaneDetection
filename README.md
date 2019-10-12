# **Finding Lane Lines on the Road**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

Repository contents
---
**Project.ipynb**       contains a Jupyter notebook with the project code and comments to be run in a browser

**findinglanes.py**     contains the same project code to be run on a local machine

**test_images**         folder containing test images

**test_videos**         folder containing test test_videos

**test_videos_out**     containing the output of the video processing pipeline


Image Processing Pipeline
---
My solution for the Lane Detection Project consists of 6 major steps:

1. The image is converted to gray scale using a helper function
2. The image is blurred with a Gaussian Filter of kernel size 5
3. In order to detect edges in the image, a canny filter with the thresholds 60/170
is applied. The thresholds were found through iteration and are in the commonly suggested 1/2 - 1/3 ratio
4. A triangular mask is defined through vertices and applied to the canny image,
leaving only the region of interest visible.
5. Using Hough transformation, line segments in the region of interest are
identified. A parameter setting of Rho = 1, Theta = Pi/180, Threshold = 50,
MinLineLength = 20 and MaxLineGap = 20 works reasonably well
6. The draw_lines function sorts the line segments based on their slope,
extrapolates a single line for both the right and left lane and draws the weighted
lanes onto the original image. Line segments over or below a slope threshold are
neglected.


Video processing
---
A video is a series of images. A video is fed image by image through the processing pipeline and saved in a output.mp4 file. 
