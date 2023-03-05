import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
from skimage.filters import threshold_local
import re
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Preprocessing part ->
# Resize the image
def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

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


def contour_to_rect(contour, resize_ratio):
    if contour is not None:
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

def bw_scanner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return (gray > T).astype("uint8") * 255

# Regex part to extracts dates and total price ->
# Multiples regex for dates to have all the formats
date_regex_patterns = [
    r"[A-Z][a-z]{2}\s\d{2}\s\d{4}",  # MMM DD YYYY
    r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
    r"\d{2}/\d{2}/\d{4}",  # DD/MM/YYYY
    r"\d{2}\.\d{2}\.\d{4}",# DD.MM.YYYY
    r"\d{2}-\d{2}-\d{4}",  # DD-MM-YYYY
    r"\d{2}\.\d{2},\d{4}", # DD-MM,YYYY
    r"\d{2}\,\d{2}.\d{4}", # DD,MM-YYYY
    r"\d{2}\,\d{2},\d{4}",  # DD,MM,YYYY
    r"\d{4},\d{2}-\d{2}",  # YYYY,MM-DD
    r"\d{4}-\d{2},\d{2}",  # YYYY-MM,DD
    r"\d{4},\d{2},\d{2}"  # YYYY,MM,DD
]

# Multiples regex for times to have all the formats
time_regex_patterns = [
    r"\d{2}:\d{2}:\d{2}", #00:00
    r"\d{2}:\d{2}"  #00:00:00
]

def getDates(text2):
    # Loop through each date regex pattern and find matches in the text for the Date time
    dates = []
    for regex in date_regex_patterns:
        matches = re.findall(regex, text2)
        if matches:
            times = matches
            break  # Stop searching if matches are found
    return dates

def getTimes(text2):
    # Loop through each date regex pattern and find matches in the text for the Date time
    times = []
    for regex in time_regex_patterns:
        matches = re.findall(regex, text2)
        if matches:
            times = matches
            break  # Stop searching if matches are found
    return times

# To get the total amount of the receipt
total_regex_patterns = [
    r"(?i)total:\s*(\w{3})?([$€£])?(\d+(?:\.\d+)?)",
    r"(?i)total\s*(\w{3})?([$€£])?(\d+(?:\.\d+)?)",
    r"(?i)Balance\s*(\w{3})?([$€£])?(\d+(?:\.\d+)?)",
    r"(?i)Balance:\s*(\w{3})?([$€£])?(\d+(?:\.\d+)?)",
    r"(?i)Debit\s*(\w{3})?([$€£])?(\d+(?:\.\d+)?)",
    r"(?i)Debit:\s*(\w{3})?([$€£])?(\d+(?:\.\d+)?)"
]

def getTotal(text2):
    # Loop through each date regex pattern and find matches in the text for the total amount
    total = []
    for regex in total_regex_patterns:
        matches = re.findall(regex, text2)
        if matches:
            total = matches
            break  # Stop searching if matches are found
    return total

# To get the total amount of the receipt
subTotal_regex_patterns = [
    r"(?i)subtotal:\s*(\w{3})?([$€£])?(\d+(?:\.\d+)?)",
    r"(?i)subtotal\s*(\w{3})?([$€£])?(\d+(?:\.\d+)?)",
    r"(?i)sous-total:\s*(\w{3})?([$€£])?(\d+(?:\.\d+)?)",
    r"(?i)sous-total\s*(\w{3})?([$€£])?(\d+(?:\.\d+)?)",
    r"(?i)Tax\s*(\w{3})?([$€£])?(\d+(?:\.\d+)?)",
    r"(?i)Tax:\s*(\w{3})?([$€£])?(\d+(?:\.\d+)?)",
]

def getSubtotal(text2):
    # Loop through each date regex pattern and find matches in the text for the total amount
    subtotal = []
    for regex in subTotal_regex_patterns:
        matches = re.findall(regex, text2)
        if matches:
            subtotal = matches
            break  # Stop searching if matches are found
    return subtotal

# To get the type of the receipt(Credit, ...)
type_regex_pattern = r'^(Cash|Credit|Debit|Prepaid|ElectronicBenefitsTransfer\(EBT\)|GiftCard|StoreCredit|MobilePayment|OnlinePayment|Check|MoneyOrder|BankTransfer|Cryptocurrency)$'

def getType(text2):
    types = re.findall(type_regex_pattern, text2, flags=re.IGNORECASE)
    return types

# File path for brands
file_path = "brands.txt"
try:
    # Read the file into a list of strings
    with open(file_path, "r", encoding="utf-8") as file:
        brands_list = [line.strip() for line in file]
except FileNotFoundError:
    print("Error: File not found")
    exit()

def getBrand(lines):
    brandName = None
    # Check the 8 first lines
    if(len(lines) > 0):
        for idx in range(1,8):
            # Loop through the list of brands and search for each brand in the sentence
            for brand in brands_list:
                pattern = re.compile(r'\b{}\b'.format(brand))
                if pattern.search(lines[idx-1]+lines[idx]):
                    brandName = brand
                    break
    return brandName

def getNewTotals(text, total, subtotal): 
    amounts = re.findall(r'\d+\s*\.\s*\d{2}\b', text)
    if(len(amounts) > 0):
        floats = [float(amount.replace(" ", "")) for amount in amounts]
        sorted_amounts = sorted(floats, reverse=True)
        total.append(sorted_amounts[0])
        cpt = 1
        while(len(amounts) > cpt and sorted_amounts[cpt] != sorted_amounts[0]):
            cpt+=1
        if(len(amounts) > cpt and sorted_amounts[cpt] != sorted_amounts[0]):
            subtotal.append(sorted_amounts[cpt])
    return (total,subtotal)

def getResult(lines, dates, times, total, subtotal, types, brand):
    data = {
        'first line': lines[0] if len(lines) > 0 else None,
        'title': brand,
        'date': dates[0] if len(dates) > 0 else None,
        'time': times[0] if len(times) > 0 else None,
        'place1': lines[2] if len(lines) > 0 else None,
        'place2': lines[3] if len(lines) > 0 else None,
        'type': types[0] if len(types) > 0 else None,
        'total': total[-1] if len(total) > 0 else None,
        'subtotal': subtotal[-1] if len(subtotal) > 0 else None
    }
    return data

def main(image):
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
    edges = cv2.Canny(dilated, 100, 200, apertureSize=3)

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

    # Get 10 largest contours
    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    # Draw the largest contours on the original image
    image_largest_contours = cv2.drawContours(image.copy(), largest_contours, -1, (0, 255, 0), 3)
    
    # Get the receipt's rectangle contour
    receipt_contour = get_receipt_contour(largest_contours)

    #correct perspective
    if receipt_contour is not None:
        scanned = wrap_perspective(original.copy(), contour_to_rect(receipt_contour, resize_ratio))
    else:
        scanned = original.copy()
    
    
    # Apply bilateral filter to smooth the image
    bilateral = cv2.bilateralFilter(scanned, 9, 75, 75)
    
    result = bw_scanner(scanned)

    # Perform OCR using Tesseract
    text = pytesseract.image_to_string(result)

    # Split the text into lines for the regex part
    lines = text.splitlines()

    # Remove spaces for having a better extraction 
    # if Tesseract put some white spaces like for the dates
    text2 = text.replace(" ", "").replace("\t", "").replace("\n", "")

    total = getTotal(text2)
    subtotal = getSubtotal(text2)
    times = getTimes(text2)
    dates = getDates(text2)
    types = getType(text2)
    brand = getBrand(lines)

    # If the total is not found
    if(len(total) < 1):
        (total, subtotal) = getNewTotals(text, total, subtotal)
    
    return getResult(lines, dates, times, total, subtotal, types, brand)