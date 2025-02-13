from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
from cv2.typing import MatLike


########## DETECT 4 CORNER SQUARES #######################################

def check_is_filled(src, contour: MatLike, desired_ratio):
    # check inside part of the contour, if colored part is considerable, called them is filled
    (x, y, w, h) = cv2.boundingRect(contour)
    area = src[y: y+h, x:x+w]
    filled_ratio = cv2.countNonZero(area)*1.0 / (w*h)
    return filled_ratio > desired_ratio

def between(num, num1, num2):
    if num1 >= num2:
        return False 
    return num >= num1 and num <= num2 

def is_big_square(contour: MatLike):
    peri = cv2.arcLength(contour,True) #extract perimeter of contour
    poly = cv2.approxPolyDP(contour,peri*0.07,True) #Polygon approximation
    area = cv2.contourArea(contour) #extract area of contour
    (_,_,w,h) = cv2.boundingRect(contour)
    return len(poly) == 4 and between(w,50,70) and between(h,50,70) #check witdth and height of square that around 50 to 70, which differs from small ones

def findCornerContours(img_path):
    phieu_tno = cv2.imread(img_path)
    phieu_tno = cv2.cvtColor(phieu_tno, cv2.COLOR_BGR2RGB)
    phieu_tno_gray = cv2.cvtColor(phieu_tno, cv2.COLOR_RGB2GRAY) # convert to grayscale
    blur = cv2.medianBlur(phieu_tno_gray,5) # blur to improve accuracy
    sharpen_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]) 
    sharpen = cv2.filter2D(blur,-1,sharpen_kernel) 
    thresh = cv2.threshold(sharpen,100,255,cv2.THRESH_BINARY_INV)[1] # convert image to binary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2) # decrease noise
    cnts, _ = cv2.findContours(close, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # find contours
    square = list(filter(is_big_square,cnts))
    square = [x for x in square if check_is_filled(thresh,x,0.85)] #filter square that was colored more than 85 percent

    return square 

#********************
def drawCornerContours(img_path):

    #visualize the detected squares
    phieu_tno = cv2.imread(img_path)
    square = findCornerContours(img_path)
    cv2.drawContours(phieu_tno, square, -1, (0,255,0), 10)
    plt.imshow(phieu_tno)
    return len(square)
#*********************


################ FIX PERSPECTIVE OF IMAGE BASED ON 4 CORNER SQUARES ####################################


def get_topleft_points(img_path):
    # get top-left points of each corner squares
    square = findCornerContours(img_path)
    lst_coor = []
    dic = dict()
    avg = 0
    for cnt in square:
        (x, y, w, h) = cv2.boundingRect(cnt)
        lst_coor.append((x,y))
        dic[(x,y)] = (w,h)

    return lst_coor, dic

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "int32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return_rect = [tuple(re) for re in list(rect)]
    
    return return_rect


def get_corner_points(img_path):
    # To take the full fixed-perspective picture with full inner square, I take: 
    # top-left coordinates of top-left square
    # top-right coordinates of top-right square
    # bottom-right coordinates of bottom-right square
    # bottom-left coordinates of bottom-left square

    lst_coor, dic = get_topleft_points(img_path)
    rect = order_points(np.array(lst_coor))

    return_rect = np.zeros((4, 2), dtype = "float32") 
    
    top_left = rect[0]
    top_right = (rect[1][0] + dic[rect[1]][0], rect[1][1])
    bottom_right = (rect[2][0] + dic[rect[2]][0], rect[2][1] + dic[rect[2]][1])
    bottom_left = (rect[3][0], rect[3][1] + dic[rect[3]][1])

    return_rect[0] = top_left
    return_rect[1] = top_right
    return_rect[2] = bottom_right
    return_rect[3] = bottom_left
    return return_rect

def four_point_transform(img_path):
	# obtain a consistent order of the points and unpack them
	# individually
    image = cv2.imread(img_path)
    
    rect = get_corner_points(img_path)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


######### SEGMENTAION #################################################################\

def is_square(contour: MatLike):       #differ from is_big_square function
    peri = cv2.arcLength(contour, True)
    poly = cv2.approxPolyDP(contour, peri*0.07, True)
    area = cv2.contourArea(contour)
    (_,_,w,h) = cv2.boundingRect(contour)
    return len(poly) == 4 and between(w, 25, 35) and between(h, 25, 35) #range for width and height of 27 inner square

def findSquareContours(image):
    #detect 27 inner squares
    phieu_tno = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    phieu_tno_gray = cv2.cvtColor(phieu_tno, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(phieu_tno_gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1],[-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    thresh = cv2.threshold(sharpen, 100, 255, cv2.THRESH_BINARY_INV)[1] #convert to binary image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) 
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2) #decrease noise
    cnts, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #find contours
    square = list(filter(is_square, cnts))
    square = [x for x in square if check_is_filled(thresh, x, 0.79)] #filter square that was colored more than 85percent
    
    return square 

def drawSquareContours(image):
    square = findSquareContours(image)
    image_copy = image.copy()
    cv2.drawContours(image_copy, square, -1, (0,255,0), 10)
    plt.imshow(image_copy)
    return len(square)


def get_centroid(contour):
    # detect centroid of each each square
    (x, y, w, h) = cv2.boundingRect(contour)

    cx = x + w//2
    cy = y + h//2 

    return (cx, cy)

def findSquareContourCentroid(image):
    square = findSquareContours(image)

    centroids = [get_centroid(contour) for contour in square]
    centroids = [c for c in centroids if c is not None]

    return centroids

def sort_points_grid(pts, row_threshold=10):
    """
    Sort a set of points from left-to-right, top-to-bottom.
    
    Args:
        pts (array-like): List or numpy array of points with shape (N, 2).
        row_threshold (int): Maximum difference in y-coordinates to group points into rows.
    
    Returns:
        sorted_points (list): Points sorted left-to-right, top-to-bottom.
    """
    # Convert points to numpy array if not already
    pts = np.array(pts)
    
    # Sort points by y-coordinate (top-to-bottom)
    pts = pts[pts[:, 1].argsort()]
    
    # Initialize list for grouped rows
    rows = []
    current_row = [pts[0]]

    # Group points into rows based on the row_threshold
    for i in range(1, len(pts)):
        if abs(pts[i, 1] - current_row[-1][1]) < row_threshold:
            current_row.append(pts[i])
        else:
            # Sort current row by x-coordinate (left-to-right)
            current_row = sorted(current_row, key=lambda pt: pt[0])
            rows.append(current_row)  # Add the sorted row to rows
            current_row = [pts[i]]  # Start a new row
    
    # Add the last row
    if current_row:
        current_row = sorted(current_row, key=lambda pt: pt[0])
        rows.append(current_row)
    
    # Combine all rows into a single list
    sorted_points = [tuple(pt) for row in rows for pt in row]
    
    return sorted_points

def segmentation_6_parts(warped, target_size=(640, 640)):

    centroids = findSquareContourCentroid(warped)
    sorted_points = sort_points_grid(centroids, row_threshold=15)
    image = warped.copy()
    # Provided pivot points (yellow points) 
    pivot1 = sorted_points[6]  
    pivot2 = sorted_points[17]  
    
    # Calculate the mid-point of the two pivot points
    mid_x = pivot1[0]
    
    # Get image dimensions
    #width, height, _ = image.shape
    height, width, _ = image.shape
    
    # Define the boundaries for the six segments
    segments = [
        (0, pivot1[1], 0, mid_x),              # Top-left
        (0, pivot1[1], mid_x, width),          # Top-right
        (pivot1[1], pivot2[1], 0, mid_x),      # Middle-left
        (pivot1[1], pivot2[1], mid_x, width),  # Middle-right
        (pivot2[1], height, 0, mid_x),         # Bottom-left
        (pivot2[1], height, mid_x, width)      # Bottom-right
    ]
    cropped_images = []
    for idx, (y1, y2, x1, x2) in enumerate(segments):
        cropped = image[y1:y2, x1:x2]
        #resized_image = cropped
        resized_image = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        cropped_images.append(resized_image)
    return cropped_images

######################################12/20/2024#################
#############segment into 4 part###################
# def segmentation_first_small_part_fortest(warped):
#     centroids = findSquareContourCentroid(warped)
#     sorted_points = sort_points_grid(centroids, row_threshold=15)

#     image = warped.copy()

#     pivot1 = sorted_points[0]
#     #pivot2 = sorted_points[1]
#     pivot3 = sorted_points[2]
#     pivot4 = sorted_points[3]

#     #width, height, _ = image.shape
#     height, width, _ = image.shape

#     pivot1 = (pivot1[0], 0)
#     pivot4 = (width, pivot4[1])

#     first_small_part = image[pivot1[1]:pivot3[1], pivot3[0]:pivot4[0]]

#     return first_small_part

# def segmentation_3_parts_fortest(warped):
#     centroids = findSquareContourCentroid(warped)
#     sorted_points = sort_points_grid(centroids, row_threshold=15)

#     image = warped.copy()

#     pivot1 = sorted_points[4] #5th
#     pivot2 = sorted_points[7] #8th
#     pivot3 = sorted_points[8] #9th
#     pivot4 = sorted_points[12] #13th

#     pivot5 = sorted_points[13] #14th
#     pivot6 = sorted_points[21] #22th
#     pivot7 = sorted_points[22] #23th
#     pivot8 = sorted_points[26] #27th 

#     height, width, _ = image.shape 

#     segments = [
#         (pivot1[1], pivot4[1], pivot1[0], pivot4[0]), #part I
#         (pivot3[1], pivot6[1], pivot3[0], pivot6[0]), #part II
#         (pivot5[1], pivot8[1], pivot5[0], pivot8[0]), #part III
#     ]

#     cropped_images = []
#     for idx, (y1, y2, x1, x2) in enumerate(segments):
#         cropped = image[y1:y2, x1:x2]
#         #resize_image = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
#         cropped_images.append(cropped)
    
#     return cropped_images

def segmentation_first_small_part(warped,sorted_points):

    image = warped.copy()

    pivot1 = sorted_points[0]
    #pivot2 = sorted_points[1]
    pivot3 = sorted_points[2]
    pivot4 = sorted_points[3]

    #width, height, _ = image.shape
    height, width, _ = image.shape

    pivot1 = (pivot1[0], 0)
    pivot4 = (width, pivot4[1])

    first_small_part = image[pivot1[1]:pivot3[1], pivot3[0]:pivot4[0]]

    return first_small_part


def segmentation_3_parts(warped, sorted_points):

    image = warped.copy()

    pivot1 = sorted_points[4] #5th
    pivot2 = sorted_points[7] #8th
    pivot3 = sorted_points[8] #9th
    pivot4 = sorted_points[12] #13th

    pivot5 = sorted_points[13] #14th
    pivot6 = sorted_points[21] #22th
    pivot7 = sorted_points[22] #23th
    pivot8 = sorted_points[26] #27th 

    height, width, _ = image.shape 

    segments = [
        (pivot1[1], pivot4[1], pivot1[0], pivot4[0]), #part I
        (pivot3[1], pivot6[1], pivot3[0], pivot6[0]), #part II
        (pivot5[1], pivot8[1], pivot5[0], pivot8[0]), #part III
    ]

    cropped_images = []
    for idx, (y1, y2, x1, x2) in enumerate(segments):
        cropped = image[y1:y2, x1:x2]
        #resize_image = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        cropped_images.append(cropped)
    
    return cropped_images

def segment_type_ii(warped): #segment based on part of question, no resized
    centroids = findSquareContourCentroid(warped)
    sorted_points = sort_points_grid(centroids, row_threshold=15) 

    first_small_part = segmentation_first_small_part(warped, sorted_points)

    cropped_images = segmentation_3_parts(warped, sorted_points)

    cropped_images.append(first_small_part)

    return cropped_images 


#############################12/22/2024##################
##########################take info of coordinates to fixed label file##########################
# def segmentation_first_small_part_info(img, sorted_points):

#     image = img.copy()

#     pivot1 = sorted_points[0]
#     #pivot2 = sorted_points[1]
#     pivot3 = sorted_points[2]
#     pivot4 = sorted_points[3]

#     #width, height, _ = image.shape
#     height, width, _ = image.shape

#     pivot1 = (pivot1[0], 0)
#     pivot4 = (width, pivot4[1])

#     first_small_part = image[pivot1[1]:pivot3[1], pivot3[0]:pivot4[0]]

#     return first_small_part , (int(pivot3[0]),int(pivot1[1]), int(pivot4[0]), int(pivot3[1]))      #info: (x_min, y_min, x_max,, y_max)

# def segmenation_3_parts_info(img, sorted_points):
#     image = img.copy()

#     pivot1 = sorted_points[4] #5th
#     pivot2 = sorted_points[7] #8th
#     pivot3 = sorted_points[8] #9th
#     pivot4 = sorted_points[12] #13th

#     pivot5 = sorted_points[13] #14th
#     pivot6 = sorted_points[21] #22th
#     pivot7 = sorted_points[22] #23th
#     pivot8 = sorted_points[26] #27th 

#     height, width, _ = image.shape 

#     segments = [
#         (pivot1[1], pivot4[1], pivot1[0], pivot4[0]), #part I
#         (pivot3[1], pivot6[1], pivot3[0], pivot6[0]), #part II
#         (pivot5[1], pivot8[1], pivot5[0], pivot8[0]), #part III
#     ]

#     cropped_images = []
#     cropped_info = []
#     for idx, (y1, y2, x1, x2) in enumerate(segments):
#         cropped = image[y1:y2, x1:x2]
#         #resize_image = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
#         cropped_images.append(cropped)
#         cropped_info.append((int(x1), int(y1), int(x2), int(y2)))

        
    
#    return cropped_images, cropped_info

def segmentation_first_small_part_info(img, sorted_points):

    image = img.copy()

    pivot1 = sorted_points[0]
    #pivot2 = sorted_points[1]
    pivot3 = sorted_points[2]
    pivot4 = sorted_points[3]

    #width, height, _ = image.shape
    height, width, _ = image.shape

    #pivot1 = (pivot1[0], 0)
    pivot4 = (width, pivot4[1])

    first_small_part = image[pivot1[1]:pivot3[1], pivot3[0]:pivot4[0]]

    return first_small_part , (int(pivot3[0]),int(pivot1[1]), int(pivot4[0]), int(pivot3[1]))      #info: (x_min, y_min, x_max,, y_max)

# def segmenation_3_parts_info(img, sorted_points):
#     image = img.copy()

#     pivot1 = sorted_points[4] #5th
#     pivot2 = sorted_points[7] #8th
#     pivot3 = sorted_points[8] #9th
#     pivot4 = sorted_points[12] #13th

#     pivot5 = sorted_points[13] #14th
#     pivot6 = sorted_points[21] #22th
#     pivot7 = sorted_points[22] #23th
#     pivot8 = sorted_points[26] #27th 

#     height, width, _ = image.shape 

#     segments = [
#         (pivot1[1], pivot4[1], pivot1[0], pivot4[0]), #part I
#         (pivot3[1], pivot6[1], pivot3[0], pivot6[0]), #part II
#         (pivot5[1], pivot8[1], pivot5[0], pivot8[0]), #part III
#     ]

#     cropped_images = []
#     cropped_info = []
#     for idx, (y1, y2, x1, x2) in enumerate(segments):
#         cropped = image[y1:y2, x1:x2]
#         #resize_image = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
#         cropped_images.append(cropped)
#         cropped_info.append((int(x1), int(y1), int(x2), int(y2)))


def segmenation_3_parts_info(img, sorted_points):
    image = img.copy()

    pivot5 = sorted_points[4] #5th
    pivot6 = sorted_points[5] #8th
    pivot9 = sorted_points[8] #9th
    pivot10 = sorted_points[9] #13th

    pivot14 = sorted_points[13] #14th
    pivot16 = sorted_points[15] #22th
    pivot17 = sorted_points[16] #23th
    pivot19 = sorted_points[18] #27th 

    pivot24 = sorted_points[23]
    pivot26 = sorted_points[25]

    height, width, _ = image.shape 

    segments = [
        (pivot5[1], pivot10[1], pivot5[0], pivot10[0]), #part I
        (pivot9[1], pivot16[1], pivot9[0], pivot16[0]), #part II
        (pivot17[1], pivot26[1], pivot17[0], pivot26[0]), #part III
    ]

    cropped_images = []
    cropped_info = []
    for idx, (y1, y2, x1, x2) in enumerate(segments):
        cropped = image[y1:y2, x1:x2]
        #resize_image = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        cropped_images.append(cropped)
        cropped_info.append((int(x1), int(y1), int(x2), int(y2)))
    
    return cropped_images, cropped_info

###################12/23/2024#####################################
####################function for test#############################
def segmentation_test_set(image):
    centroids = findSquareContourCentroid(image)
    sorted_points = sort_points_grid(centroids, row_threshold=15)

    height, width, _ = image.shape

    segments = [
        (sorted_points[0][1], sorted_points[3][1], sorted_points[0][0], width),
        (sorted_points[4][1], sorted_points[9][1], sorted_points[4][0], sorted_points[9][0]),
        (sorted_points[5][1], sorted_points[10][1], sorted_points[5][0], sorted_points[10][0]),
        (sorted_points[6][1], sorted_points[11][1], sorted_points[6][0], sorted_points[11][0]),
        (sorted_points[7][1], sorted_points[11][1], sorted_points[11][0], sorted_points[7][0]),
        (sorted_points[8][1], sorted_points[15][1], sorted_points[8][0], sorted_points[15][0]),
        (sorted_points[9][1], sorted_points[17][1], sorted_points[9][0], sorted_points[17][0]),
        (sorted_points[10][1], sorted_points[19][1], sorted_points[10][0], sorted_points[19][0]),
        (sorted_points[11][1], sorted_points[21][1], sorted_points[11][0], sorted_points[21][0]),
        (sorted_points[13][1], sorted_points[23][1], sorted_points[13][0], sorted_points[23][0]),
        (sorted_points[16][1], sorted_points[25][1], sorted_points[16][0], sorted_points[25][0]),
        (sorted_points[21][1], sorted_points[25][1], sorted_points[25][0], sorted_points[21][0]),
    ]

    cropped_images = []
    cropped_info = []
    for idx, (y1, y2, x1, x2) in enumerate(segments):
        cropped = image[y1:y2, x1:x2]
        #resize_image = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        cropped_images.append(cropped)
        cropped_info.append((int(x1), int(y1), int(x2), int(y2)))
    
    return cropped_images, cropped_info

