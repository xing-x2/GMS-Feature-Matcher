import numpy as np
from enum import Enum
import time
import cv2
from cv2.xfeatures2d import matchGMS
from cv2.xfeatures2d import SIFT_create
from compare import find_ground_truth
import argparse



class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2
    COLOR_CODED_POINTS_X = 3
    COLOR_CODED_POINTS_Y = 4
    COLOR_CODED_POINTS_XpY = 5


def draw_matches(src1, src2, kp1, kp2, matches, drawing_type):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    if drawing_type == DrawingType.ONLY_LINES:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))

    elif drawing_type == DrawingType.LINES_AND_POINTS:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

    elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))

            if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                colormap_idx = int(left[0] * 256. / src1.shape[1])  # x-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                colormap_idx = int(left[1] * 256. / src1.shape[0])  # y-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                colormap_idx = int((left[0] - src1.shape[1]*.5 + left[1] - src1.shape[0]*.5) * 256. / (src1.shape[0]*.5 + src1.shape[1]*.5))  # manhattan gradient

            color = tuple(map(int, _colormap[colormap_idx, 0, :]))
            cv2.circle(output, tuple(map(int, left)), 1, color, 2)
            cv2.circle(output, tuple(map(int, right)), 1, color, 2)
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--im2", required=True)
    opt = p.parse_args()
    if opt.name == "boat":
        ext = "pgm"
    else:
        ext = "ppm"
    path1 = "GT_pics/"+opt.name+"/imgs/img1."+ext
    path2 = "GT_pics/"+opt.name+"/imgs/img"+opt.im2+"."+ext
    gt_path = "ground_truth/"+opt.name+"/"+opt.name+"_1_"+opt.im2+"_TP.txt"

    img1 = cv2.imread(path1)# ("../data/01.jpg")
    img2 = cv2.imread(path2)# ("../data/02.jpg")

    orb = cv2.ORB_create(10000)
    orb.setFastThreshold(0)
    # detector = cv2.xfeatures2d.SIFT_create(nfeatures=8000, contrastThreshold=1e-5)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_all = matcher.match(des1, des2)

    start = time.time()
    matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches_all, withScale=False, withRotation=False, thresholdFactor=6)
    end = time.time()
    # print(len(matches_gms))

    kin1 = []
    kin2 = []
    for match in matches_gms:
        i1 = match.queryIdx
        i2 = match.trainIdx
        kin1.append(kp1[i1].pt)
        kin2.append(kp2[i2].pt)
    kin1 = np.array(kin1)
    kin2 = np.array(kin2)
    true_pos, false_pos = find_ground_truth(kin1, kin2, gt_path)


    print('Found', len(matches_gms), 'matches')
    print('GMS takes', end-start, 'seconds')

    output = draw_matches(img1, img2, kp1, kp2, matches_gms, DrawingType.ONLY_LINES)

    cv2.imshow("show", output)
    cv2.waitKey(0)
