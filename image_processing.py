import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import affinity_propagation
from matplotlib import pyplot as plt






def get_coords(hMin, sMin, vMin, hMax, sMax, vMax, image):
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    kernel = np.ones((20, 20), np.uint8)
    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    eroded = cv2.erode(mask, kernel)
    labeled = ndimage.label(mask)
    counts = np.unique(labeled[0], return_counts=True)
    counts_dict = dict(zip(counts[0], counts[1]))
    counts_dict = {key:val for key, val in counts_dict.items() if val > 500}
    for item in counts[0]:
        if item not in counts_dict.keys():
            labeled[0][labeled[0] == item] = 0
    # cv2.bitwise_and(image, image, mask=mask)
    print(counts)
    large_objects = np.unique(cv2.bitwise_and(labeled[0], labeled[0], mask=eroded))
    eroded_labeled = np.copy(labeled[0])
    eroded_labeled_bool = np.copy(labeled[0])
    for item in counts[0]:
        if item not in large_objects:
            eroded_labeled[eroded_labeled == item] = 0
            eroded_labeled_bool[eroded_labeled_bool == item] = 0
        else:
            eroded_labeled_bool[eroded_labeled_bool == item] = 1
    # Display result image
    # plt.imshow(image)
    # plt.show()
    # plt.imshow(labeled[0])
    # plt.show()
    # print(large_objects)
    return ndimage.measurements.center_of_mass(eroded_labeled_bool, eroded_labeled, large_objects[1:])

