import sys
import numpy as np
from scipy import stats
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.stats import chi2
import cv2
CONFIDENCE = 0.95
NUM_OF_INTERVALS = 25


def rgb2gray(img):
    return np.dot(img[:255, :3], [0.299, 0.587, 0.144])


def median(arr):
    sum_of_freq = []
    s, i, x0 = 0, 0, 0
    for a in arr[0]:
        s += a
        sum_of_freq.append(s)
        if sum_of_freq[i] >= sum(arr[0]) / 2:
            x0 = arr[1][i]
            break
        else:
            i += 1
    h = arr[1][1] - arr[1][0]
    m = x0 + h * ((sum(arr[0]) / 2) - sum_of_freq[len(sum_of_freq) - 2]) / (
                sum_of_freq[len(sum_of_freq) - 2] - sum_of_freq[len(sum_of_freq) - 3])
    return m


def pirson(hist, img):
    n_i = list(hist[0])
    x_i, p_i, k = [], [], []
    X = np.mean(img.ravel())
    S = np.std(img.ravel())

    for i in range(0, NUM_OF_INTERVALS):
        x_i.append((hist[1][i] + hist[1][i + 1]) / 2)
    for i in range(0, NUM_OF_INTERVALS - 1):
        p_i.append((stats.norm.cdf((x_i[i + 1] - X) / S) - 0.5) - (
                    stats.norm.cdf((x_i[i] - X) / S) - 0.5))
    p_i.append(0.5 - (stats.norm.cdf((x_i[NUM_OF_INTERVALS - 1] - X) / S) - 0.5))
    n = len(img.ravel())
    for i in range(0, NUM_OF_INTERVALS):
        k.append(np.square(n_i[i] - n * p_i[i]) / (n * p_i[i]))
    return sum(k), chi2.isf(1 - CONFIDENCE, NUM_OF_INTERVALS - 3)


def main():
    sea_img = mpimg.imread('sea.jpg')
    panorama_img = mpimg.imread('sky.jpg')
    sea = cv2.imread('sea.jpg', 0)
    sea = cv2.resize(sea, (800, 800))
    cv2.imshow("Sea", sea)
    cv2.waitKey(0)
    panorama = cv2.imread('sky.jpg', 0)
    panorama = cv2.resize(panorama, (800, 800))
    cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)
    sea_gray_img = rgb2gray(sea_img)
    panorama_gray_img = rgb2gray(panorama_img)
    sea_hist = plt.hist(sea_gray_img.ravel(), bins=NUM_OF_INTERVALS, label="sea histogram")
    plt.show()
    panorama_hist = plt.hist(panorama_gray_img.ravel(), bins=NUM_OF_INTERVALS, label="panorama histogram")
    plt.show()
    print(f'Sea mean {np.mean(sea_gray_img.ravel())}')
    print(f'Panorama mean {np.mean(panorama_gray_img.ravel())}')
    print(f'Sea std {np.std(sea_gray_img.ravel())}')
    print(f'Panorama std {np.std(panorama_gray_img.ravel())}')
    ind1 = np.where(sea_hist[0] == max(sea_hist[0]))[0][0]
    ind2 = np.where(panorama_hist[0] == max(panorama_hist[0]))[0]
    print(f'Sea hist mode interval ({sea_hist[1][ind1]}, {sea_hist[1][ind1 + 1]})')
    print(f'Panorama hist mode intervals ({panorama_hist[1][ind2]}, {panorama_hist[1][ind2 + 1]}')
    print(f'Sea hist median {median(sea_hist)}')
    print(f'Panorama hist median {median(panorama_hist)}')
    print(f'Corrcoef of hist {np.corrcoef(sea_hist[0], panorama_hist[0])[0][1]}')
    print(f'Corrcoef of images {np.corrcoef(sea_gray_img.ravel(),panorama_gray_img.ravel())[0][1]}')
    sea_k_lookable, sea_k_kr = pirson(sea_hist, sea_gray_img)
    print(f'K_lookable = {sea_k_lookable}, K_kr = {sea_k_kr}')
    if sea_k_lookable < sea_k_kr:
        print('the null hypothesis is true for sea image')
    else:
        print('the null hypothesis rejected for sea image')

    panorama_k_lookable, panorama_k_kr = pirson(panorama_hist, panorama_gray_img)
    print(f'K_lookable = {panorama_k_lookable}, K_kr = {panorama_k_kr}')
    if panorama_k_lookable < panorama_k_kr:
        print('the null hypothesis is true for sky image')
    else:
        print('the null hypothesis rejected for sky image')


if __name__ == '__main__':
    main()
    sys.exit(0)