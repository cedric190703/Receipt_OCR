import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
from skimage.filters import threshold_local
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Load the image using OpenCV
image = cv2.imread(r'C:\Users\cbrzy\OneDrive\Bureau\ticket.jpg') 

#Preprocessing part ->
#resize the image
def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

resize_ratio = 500 / image.shape[0]
original = image.copy()
image = opencv_resize(image, resize_ratio)

# Grayscale filter
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blurred filter
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Dilate the image to enhance edges
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
dilated = cv2.dilate(blurred, rectKernel)

# Apply Canny edge detection to detect edges
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find contours in the image using the edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a list to store squares
squares = []

# Iterate through the contours to find rectangles
for contour in contours:
    # Approximate the contour to a polygon
    approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True)*0.02, True)
    
    # If the polygon has 4 vertices, we consider it a rectangle
    if len(approx) == 4:
        squares.append(approx)

# Sort the squares by area in descending order
squares = sorted(squares, key=cv2.contourArea, reverse=True)

# Draw the largest square on the original image
if len(squares) > 0:
    largest_square = squares[0]
    cv2.drawContours(image, largest_square, 0, (0,0,255), 3)

# Display the results
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Get 10 largest contours
largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

# Draw the largest contours on the original image
image_largest_contours = cv2.drawContours(image.copy(), largest_contours, -1, (0, 255, 0), 3)

# Approximate the contour by a more primitive polygon shape
def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.032 * peri, True)

def get_receipt_contour(contours):    
    # Loop over the contours
    for c in contours:
        approx = approximate_contour(c)
        # If our approximated contour has four points, we can assume it is receipt's rectangle
        if len(approx) == 4:
            return approx

# Get the receipt's rectangle contour
receipt_contour = get_receipt_contour(largest_contours)

def contour_to_rect(contour):
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    # top-left point has the smallest sum
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points:
    # the top-right will have the minumum difference 
    # the bottom-left will have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect / resize_ratio

def wrap_perspective(img, rect):
    # unpack rectangle points: top left, top right, bottom right, bottom left
    (tl, tr, br, bl) = rect
    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # destination points which will be used to map the screen to a "scanned" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # warp the perspective to grab the screen
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

scanned = wrap_perspective(original.copy(), contour_to_rect(receipt_contour))

# Apply bilateral filter to smooth the image
bilateral = cv2.bilateralFilter(scanned, 9, 75, 75)

# Invert the thresholded image
inverted = cv2.bitwise_not(bilateral)

# Apply fixed-level binary threshold to the inverted image
threshold_value = 78
_, result = cv2.threshold(inverted, threshold_value, 255, cv2.THRESH_BINARY)

# Perform OCR using Tesseract
text = pytesseract.image_to_string(result)

# Print the extracted text
print(text)

d = pytesseract.image_to_data(result, output_type=Output.DICT)
n_boxes = len(d['level'])
boxes = cv2.cvtColor(result.copy(), cv2.COLOR_BGR2RGB)
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])    
    boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("boxes", boxes)

extracted_text = pytesseract.image_to_string(image)
print(extracted_text)
cv2.imshow('image result',result)
cv2.waitKey(0)