import cv2
import numpy as np
import math
from PIL import Image
import PIL.ImageOps
import os

def validate_contour(contour, img, aspect_ratio_range, area_range):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width = rect[1][0]
    height = rect[1][1]
    output = False

    if width > 0 and height > 0:
        aspect_ratio = float(width) / height if width > height else float(height) / width

        if (aspect_ratio >= aspect_ratio_range[0] and aspect_ratio <= aspect_ratio_range[1]):
            if ((height * width > area_range[0] ) and (height * width < area_range[1])):
                box_copy = list(box)
                point = box_copy[0]
                del (box_copy[0])
                dists = [((p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2) for p in box_copy]
                sorted_dists = sorted(dists)
                opposite_point = box_copy[dists.index(sorted_dists[1])]
                tmp_angle = 90

                if abs(point[0] - opposite_point[0]) > 0:
                    tmp_angle = abs(float(point[1] - opposite_point[1])) / abs(point[0] - opposite_point[0])
                    tmp_angle = rad_to_deg(math.atan(tmp_angle))

                if tmp_angle <= 45:
                    output = True

    return output


def deg_to_rad(angle):
    return angle * np.pi / 180.0


def rad_to_deg(angle):
    return angle * 180 / np.pi


def enhance(img):
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [1, 0, 1]])
    return cv2.filter2D(img, -1, kernel)


def process_image(path, i, debug, **options):
    # choosing the kernel for morphological operations
    if options.get('type') == 'rect':
        se_shape = (12, 4)

    elif options.get('type') == 'square':
        se_shape = (7, 6)

    image = Image.open(path)
    inverted_image = PIL.ImageOps.invert(image)
    inverted_path = './images/inverted/'+str(i)+".jpg"
    inverted_image.save(inverted_path)

    raw_image = cv2.imread(inverted_path)
    input_image = np.copy(raw_image)
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray = enhance(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.Sobel(gray, -1, 1, 0)
    h, sobel = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('Nakon threshold binary', sobel)
    #cv2.waitKey(0)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, se_shape)
    gray = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, se)

    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('Nakon morfološke operacije', gray)
    #cv2.waitKey(0)

    ed_img = np.copy(gray)
    _, contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        aspect_ratio_range = (2.2, 12)
        area_range = (0, 15000)
        if options.get('type') == 'rect':
            aspect_ratio_range = (1.2, 12)
            area_range = (2000, 15000)

        elif options.get('type') == 'square':
            aspect_ratio_range = (1, 2)
            area_range = (300, 8000)

        if validate_contour(contour, gray, aspect_ratio_range, area_range):
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)
            angle = rect[2]

            if angle < -45:
                angle += 90

            W = rect[1][0]
            H = rect[1][1]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            size = (x2 - x1, y2 - y1)
            M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
            tmp = cv2.getRectSubPix(ed_img, size, center)
            tmp = cv2.warpAffine(tmp, M, size)
            TmpW = H if H > W else W
            TmpH = H if H < W else W
            tmp = cv2.getRectSubPix(tmp, (int(TmpW), int(TmpH)), (size[0] / 2, size[1] / 2))
            __, tmp = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_pixels = 0

            for x in range(tmp.shape[0]):
                for y in range(tmp.shape[1]):
                    if tmp[x][y] == 255:
                        white_pixels += 1

            edge_density = float(white_pixels) / (tmp.shape[0] * tmp.shape[1])

            if edge_density > 0.5:
                cv2.drawContours(input_image, [box], 0, (127, 0, 255), 2)
                original = cv2.imread(path)
                registration = original[y1:y2, x1:x2]
                winPath = "./images/registration/" + str(i)
                cv2.imwrite(winPath + ".png", registration)

    return input_image


for i in range(1, 20):
    path = './images/' + str(i) + '.jpg'
    if os.path.exists(path) == False:
        continue
    o1 = process_image(path, i, 3, type='rect')
    cv2.imshow('detected'+str(o1.shape), o1)
    cv2.waitKey(0)

cv2.destroyAllWindows()