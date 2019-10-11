#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from PIL import Image
from moviepy.editor import VideoFileClip

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
#print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image)

def grayscale(img):

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """Applies an image mask."""
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # sort lines into right and left lane
    right_lane, left_lane = sort_lines(lines)

    #extrapolate and draw lines on image
    rX1, rY1, rX2, rY2 = extrapolate_lines(right_lane)
    lX1, lY1, lX2, lY2 = extrapolate_lines(left_lane)

    cv2.line(line_img, (rX1, rY1), (rX2, rY2), [255, 0, 0], 5)
    cv2.line(line_img, (lX1, lY1), (lX2, lY2), [255, 0, 0], 5)

    return weighted_img(line_img, img)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):

    return cv2.addWeighted(initial_img, α, img, β, γ)


def slope(x1, y1, x2, y2):
    # calculates the slope of a line segment
    return (y1- y2)/(x1- x2)


def sort_lines(lines, mmax = 3, mmin = 0.5):
    #sorts line segments based on their slope

    right_lane = []
    left_lane = []

    for x1,y1,x2,y2 in lines[:,0]:
        m = slope(x1,y1, x2, y2)
        if (abs(m) > mmax or abs(m) < mmin):
            continue
        elif(m >= 0):
            right_lane.append([x1, y1, x2, y2])
        else:
            left_lane.append([x1, y1, x2, y2])

    return right_lane, left_lane


def extrapolate_lines(lines):
    # extrapolates an array of line segments to a single line with fixed y coordinates

    x = []
    y = []

    for x1, y1, x2, y2 in lines:
        x += [x1, x2]
        y += [y1, y2]

    line = np.polyfit(x, y, 1)
    m, b = line

    Y1 = int(540)
    Y2 = int(540 * (3/5))
    X1 = int((Y1-b) / m)
    X2 = int((Y2-b) / m)

    return X1, Y1, X2, Y2




###PIPELINE###
image = mpimg.imread('test_images/solidWhiteRight.jpg')

imshape = image.shape

# transform image to grayscale
gray = grayscale(image)

# apply blurring
blurred = gaussian_blur(gray, 5)


# aplly canny filter
low_threshold = 60
high_threshold = 170
edges = canny(blurred, low_threshold, high_threshold)


# define vertices for mask
vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)

# mask image
masked_edges = region_of_interest(edges, vertices)


# apply Hough transformation
lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, np.array([]), 20, 20)
line_img = np.zeros((imshape[0], imshape[1], 3), dtype=np.uint8)


# draw weighted lines on image
weighted_image = draw_lines(image, lines)


##Display Test Image##
#plt.imshow(weighted_image)
#plt.show()


def process_image(image):

    imshape = image.shape

    # transform image to grayscale
    gray = grayscale(image)

    # apply blurring
    blurred = gaussian_blur(gray, 5)

    # aplly canny filter
    low_threshold = 60
    high_threshold = 170
    edges = canny(blurred, low_threshold, high_threshold)

    # define vertices for mask
    vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)

    # mask image
    masked_edges = region_of_interest(edges, vertices)

    # apply Hough transformation
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, np.array([]), 20, 20)
    line_img = np.zeros((imshape[0], imshape[1], 3), dtype=np.uint8)

    # draw weighted lines on image
    weighted_image = draw_lines(image, lines)

    return weighted_image



##Display Output Video##

cap = cv2.VideoCapture("test_videos/solidWhiteRight.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    weighted_image = process_image(frame)
    cv2.imshow("putput", weighted_image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
