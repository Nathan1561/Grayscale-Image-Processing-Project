import cv2 as cv
import sys
import numpy as np
#Load the image and convert to RGB
image = 'colorcast1.jpg'
image_load = cv.imread(image)
image_rgb = cv.cvtColor(image_load, cv.COLOR_BGR2RGB)
#Displays image
def display(image_load, title="Image"):
    image_bgr = cv.cvtColor(image_load, cv.COLOR_RGB2BGR)
    cv.imshow('Image', image_bgr)
    cv.waitKey(0)
#White balance algorthim(Gray world)
def white_balance(image_rgb):
    avg_rgb = np.mean(image_rgb, axis=(0,1))
    gray_weight = avg_rgb/avg_rgb.mean()
    whitebal_gray = image_rgb * (1/gray_weight)* 0.95
    whitebal_gray = np.clip(whitebal_gray, 0, 255).astype(np.uint8)
    return whitebal_gray
#Finds the number of k clusters for seperating the image into sections
def k_means(image_rgb, max=10):
    Z = image_rgb.reshape((-1, 3))
    Z = np.float32(Z)
    values = []
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    for i in range(1, max + 1):
        compactness,_,_ = cv.kmeans(Z, i, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        values.append(compactness)
    optimal = 1
    for i in range(1, len(values) - 1):
        if(values[i - 1] - values[i]) > (values[i] - values[i + 1]):
            optimal = i + 1
            break
    return optimal
#splits the image up with k-means clustering
def image_split(image_rgb, k=8):
    Z = image_rgb.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _,labels,centers = cv.kmeans(Z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape(image_rgb.shape[:2])
    return labels, centers
#using the optimal k number, split the image
k = k_means(image_rgb, max=10)
labels, centers = image_split(image_rgb, k)
#isolate the region and apply the white balance
regions = []
for i in range(k):
    mask = np.where(labels == i, 1, 0).astype(np.uint8)
    separate_region = cv.bitwise_and(image_rgb, image_rgb, mask=mask)
    whitebal_region = white_balance(separate_region)
    regions.append(whitebal_region)
#take the white balanced sections and combine it back together
result = np.zeros_like(image_rgb)
for i, region in enumerate(regions):
    mask = np.where(labels == i, 1, 0).astype(np.uint8)
    result += region * mask[:, :, np.newaxis]
#Final display
display(image_rgb, title="Input Image")
display(result, title="White balanced image")
cv.imwrite('Result.jpg', cv.cvtColor(result, cv.COLOR_RGB2BGR))

