import cv2


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
    print(gray_sim)
    return gray_sim


def compare_color(pic1, pic2):
    img1 = cv2.imread(pic1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(pic2, cv2.IMREAD_COLOR)
    clrhist1 = cv2.calcHist([img1], [0,1,2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
    clrhist2 = cv2.calcHist([img2], [0,1,2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
    clr_sim = cv2.compareHist(clrhist1, clrhist2, method=cv2.HISTCMP_BHATTACHARYYA)
    return clr_sim


def compare_norm_color(pic1, pic2):
    # Compares normalized color histograms
    img1 = cv2.imread(pic1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(pic2, cv2.IMREAD_COLOR)

    clrhist1 = cv2.calcHist([img1], [0,1,2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
    clrhist2 = cv2.calcHist([img2], [0,1,2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(clrhist1, clrhist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(clrhist2, clrhist2, 0, 1, cv2.NORM_MINMAX)
    clr_sim = cv2.compareHist(clrhist1, clrhist2, method=cv2.HISTCMP_BHATTACHARYYA)
    return clr_sim

