import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import argparse
import time
import copy

def biggest_contour(contours, min_area):
    biggest = None
    max_area = 0
    biggest_n = 0
    approx_contour = None
    for n, i in enumerate(contours):
        area = cv2.contourArea(i)

        if area > min_area / 30:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
                biggest_n = n
                approx_contour = approx

    return biggest_n, approx_contour


def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect



def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def main(args):
    segmentation_num_pixels_x, segmentation_num_pixels_y = args.segmentation_num_pixels_x, args.segmentation_num_pixels_y
    illumination_num_pixels_x, illumination_num_pixels_y = args.illumination_num_pixels_x, args.illumination_num_pixels_y


    test_img_path = args.test_img_path

    segmentation_model_path = args.segmentation_model_path
    illumination_model_path = args.illumination_model_path

    NORM_MEAN = [ 0.485, 0.456, 0.406 ]
    NORM_STD  = [ 0.229, 0.224, 0.225 ]

    loader = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean = NORM_MEAN, std = NORM_STD)])

    start = time.time()
    ####################################
    ##### 0. Preprocess Segment ########
    ####################################

    orig_im = Image.open(f"{test_img_path}")
    orig_im = ImageOps.exif_transpose(orig_im)
    orig_y_px, orig_x_px = orig_im.size
    im = copy.deepcopy(orig_im)
    im = im.resize((segmentation_num_pixels_y, segmentation_num_pixels_x))
    im = np.array(im)

    orig_im = np.array(orig_im)

    ####################################
    ######### 1. Segmentation ##########
    ####################################
    image = loader(im)
    image = image.unsqueeze(0)   
    print(f"Inital image shape: {image.shape}")

    segmentation_model = torch.jit.load(segmentation_model_path)
    segmentation_model.eval()
    mask_det = segmentation_model(image)

    mask_to_find = mask_det[0]
    mask_to_find = np.array(mask_to_find*255).astype(np.uint8)
    mask_to_find = Image.fromarray(mask_to_find[:, :, 0])

    mask_to_find = mask_to_find.resize((orig_y_px, orig_x_px))
    mask_to_find = np.array(mask_to_find)
    contours, hierarchy = cv2.findContours(mask_to_find, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    simplified_contours = []

    for cnt in contours:
        hull = cv2.convexHull(cnt)
        simplified_contours.append(cv2.approxPolyDP(hull,
                                                    0.001 * cv2.arcLength(hull, True), True))
    biggest_n, approx_contour = biggest_contour(simplified_contours, 1000)

    warped = four_point_transform(orig_im, approx_contour)
    print(f"{time.time() - start}")


    ####################################
    ### 2. Preprocess Illumination #####
    ####################################

    warped_im = Image.fromarray(warped)
    warped_im = warped_im.resize((illumination_num_pixels_y, illumination_num_pixels_x))
    warped_im = np.array(warped_im)
    warped_im = loader(warped_im)
    warped_im = warped_im.unsqueeze(0)  
    print(f"Warped image shape: {warped_im.shape}")

    ####################################
    ######### 3. Illumination ##########
    ####################################

    illumination_model = torch.jit.load(illumination_model_path)
    illumination_model.eval()
    rectified = illumination_model(warped_im)[0]
    rectified = np.array(rectified)
    print(f"Rectified image shape: {rectified.shape}")
    print(f"{time.time() - start}")


    ####################################
    ######### 4. Post process ##########
    ####################################
    PIL_image = Image.fromarray(warped.astype('uint8'), 'RGB')
    PIL_image.save(f"{args.path_to_save_segmented_image}")

    PIL_image = Image.fromarray(rectified.astype('uint8'), 'RGB')
    PIL_image.save(f"{args.path_to_save_corrected_image}")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_img_path', type=str, default='./test.jpg', help='path to image to corrected')
    parser.add_argument('--segmentation_model_path', type=str, default='./lite_weights/segmentation_model_0409_256px_case1.ptl', help='path to segmentation model')
    parser.add_argument('--illumination_model_path', type=str, default='./lite_weights/illumination_model_0409_1600px_case1.ptl', help='path to illumination model')

    parser.add_argument('--segmentation_num_pixels_x', type=int, default=256, help='number of pixels for segmentation X')
    parser.add_argument('--segmentation_num_pixels_y', type=int, default=192, help='number of pixels for segmentation Y')

    parser.add_argument('--illumination_num_pixels_x', type=int, default=1600, help='number of pixels for segmentation X')
    parser.add_argument('--illumination_num_pixels_y', type=int, default=1200, help='number of pixels for segmentation Y')

    parser.add_argument('--path_to_save_segmented_image', type=str, default='./output_segmented.jpg', help='path to save segmented image')
    parser.add_argument('--path_to_save_corrected_image', type=str, default='./output_test.jpg', help='path to save corrected image')
    
    args = parser.parse_args()
    main(args)