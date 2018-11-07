import cv2
import os


def compare_gray(pic1, pic2):
    # img1 and img2 should both be numpy arrays at this point
    gray1 = cv2.imread(pic1, cv2.IMREAD_GRAYSCALE)
    gray2 = cv2.imread(pic2, cv2.IMREAD_GRAYSCALE)
    size1, size2 = gray1.size, gray2.size

    grayhist1 = cv2.calcHist([gray1], [0], None, [256], [0,256])
    grayhist2 = cv2.calcHist([gray2], [0], None, [256], [0,256])
    # comparing the histograms
    cv2.normalize(grayhist1, grayhist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(grayhist2, grayhist2, 0, 1, cv2.NORM_MINMAX)
    gray_sim = cv2.compareHist(grayhist1, grayhist2, method=cv2.HISTCMP_CORREL)
    print(gray_sim)
    return gray_sim


def compare_norm_gray(pic1, pic2):
    # img1 and img2 should both be numpy arrays at this point
    gray1 = cv2.imread(pic1, cv2.IMREAD_GRAYSCALE)
    gray2 = cv2.imread(pic2, cv2.IMREAD_GRAYSCALE)
    size1, size2 = gray1.size, gray2.size

    grayhist1 = cv2.calcHist([gray1], [0], None, [256], [0,256])
    grayhist2 = cv2.calcHist([gray2], [0], None, [256], [0,256])
    # comparing the histograms
    gray_sim = cv2.compareHist(grayhist1, grayhist2, method=cv2.HISTCMP_CORREL)
    return gray_sim


def compare_color(pic1, pic2):
    img1 = cv2.imread(pic1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(pic2, cv2.IMREAD_COLOR)
    clrhist1 = cv2.calcHist([img1], [0,1,2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
    clrhist2 = cv2.calcHist([img2], [0,1,2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
    clr_sim = cv2.compareHist(clrhist1, clrhist2, method=cv2.HISTCMP_BHATTACHARYYA)
    return 1-clr_sim


def compare_norm_color(pic1, pic2):
    # Compares normalized color histograms
    img1 = cv2.imread(pic1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(pic2, cv2.IMREAD_COLOR)
    print(type(img1))
    print(type(img2))
    clrhist1 = cv2.calcHist([img1], [0,1,2], None, [64,64,64], [0, 256, 0, 256, 0, 256])
    clrhist2 = cv2.calcHist([img2], [0,1,2], None, [64,64,64], [0, 256, 0, 256, 0, 256])
    cv2.normalize(clrhist1, clrhist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(clrhist2, clrhist2, 0, 1, cv2.NORM_MINMAX)
    clr_sim = cv2.compareHist(clrhist1, clrhist2, method=cv2.HISTCMP_BHATTACHARYYA)
    # should this be subtracted from 1?
    return 1-clr_sim


def compare_keypoints(pic1, pic2):
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(pic1, pic2)
    matches = sorted(matches, key=lambda x: x.distance)
    matched_img = cv2.drawMatches()
    pass


def directory_hist(dir1, dir2):
    current_path = os.getcwd()
    path1 = os.path.join(current_path, dir1)
    path2 = os.path.join(current_path, dir2)
    files1 = [f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))]
    files2 = [f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))]
    print('Files in Found1 : ' + str(files1))
    print('Files in Found2 : ' + str(files2))
    comparison_results = []
    for file1 in files1:
        pathfile1 = os.path.join(path1, file1)
        for file2 in files2:
            pathfile2 = os.path.join(path2, file2)
            comparison_results.append([compare_norm_color(pathfile1, pathfile2), file1, file2])
    return comparison_results


def directory_keypoints(dir1, dir2):
    current_path = os.getcwd()
    path1 = os.path.join(current_path, dir1)
    path2 = os.path.join(current_path, dir2)
    files1 = [f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))]
    files2 = [f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))]
    kpmatch_results = []
    for file1 in files1:
        for file2 in files2:
            kpmatch_results.append([compare_keypoints(file1, file2), file1, file2])
    return kpmatch_results
