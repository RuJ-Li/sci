import sys
import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def findContours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (5, 5))
    edged = cv2.Canny(gray_blurred, 100, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def drawContours(contours, shape, name):
    canvas = np.full(shape, 255, dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, (0, 0, 0), 1)
    cv2.imshow(name, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getColorInsideContours(img, contours):
    color_inside_contour = np.empty((len(contours), 3))

    for i, contour in enumerate(contours):
        # Calculate the mean HSV values for the current contour
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv, mask=mask)
        color_inside_contour[i] = mean_hsv[:3]

    return color_inside_contour


def clusterByHue(contours, colors, eps=1.1, min_samples=2):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(colors[:, 0].reshape(-1, 1))

    labeled_contours = {}
    labeled_colors = {}
    for i, label in enumerate(labels):
        if label not in labeled_contours:
            labeled_contours[label] = []
        if label not in labeled_colors:
            labeled_colors[label] = []

        labeled_contours[label].append(contours[i])
        labeled_colors[label].append(colors[i])

    return labeled_contours, labeled_colors


def selectModeCluster(labeled_contours):
    mode_cluster_label = 0
    mode = 0
    for label, contours in labeled_contours.items():
        if len(contours) > mode:
            mode_cluster_label = label
            mode = len(contours)

    return labeled_contours[mode_cluster_label]


def showLabels(labeled_contours, shape, labeled_colors):
    for label, contours in labeled_contours.items():
        canvas = cv2.cvtColor(np.full(shape, 255, dtype=np.uint8), cv2.COLOR_BGR2HSV)

        for i, contour in enumerate(contours):
            cv2.drawContours(canvas, [contour], -1, labeled_colors[label][i], thickness=-1)

        cv2.imshow("label: {}".format(label), cv2.cvtColor(canvas, cv2.COLOR_HSV2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def findCentroidOfContours(img, contours):
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(img, (cx, cy), 1, (0, 0, 0), -1)  # Use cv2.circle here
    return img


def isNonWhite(img, xx, xy, yx, yy):
    for i in range(xx, xy):
        for j in range(yx, yy):
            for k in range(len(img[i, j])):
                if img[i, j, k] != 255:
                    return True

    return False


def fullByBlack(img, xx, xy, yx, yy):
    for i in range(xx, xy):
        for j in range(yx, yy):
            for k in range(len(img[i, j])):
                img[i, j, k] = 0

    return img


def generateQr(filtered_contours, qr_size):
    qr = np.full(img.shape, 255, dtype=np.uint8)
    qr = findCentroidOfContours(qr, filtered_contours)

    block_x = qr.shape[0] // qr_size
    block_y = qr.shape[1] // qr_size

    for i in range(0, qr.shape[0], block_x):
        for j in range(0, qr.shape[1], block_y):
            xy = min(i + block_x, qr.shape[0])
            yy = min(j + block_y, qr.shape[1])
            if isNonWhite(qr, i, xy, j, yy):
                fullByBlack(qr, i, xy, j, yy)

    return qr


if __name__ == '__main__':
    img_path, qr_path = sys.argv[1:]

    img = cv2.imread(img_path)
    cv2.imshow("raw img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Call the findContours function to get the contours
    contours = findContours(img)

    # draw raw contours
    drawContours(contours, img.shape, "raw contours")

    # find mean hsv of each contour
    color_inside_contour = getColorInsideContours(img, contours)

    # cluster based on hue
    labeled_contours, labeled_colors = clusterByHue(contours, color_inside_contour)

    # show each cluster
    showLabels(labeled_contours, img.shape, labeled_colors)

    # select the mode group
    filtered_contours = selectModeCluster(labeled_contours)

    # draw filter contours
    drawContours(filtered_contours, img.shape, "filtered contours")

    # convert filtered contours into black blocks to form a qr image
    qr = generateQr(filtered_contours, 32)

    cv2.imshow("qr", qr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(qr_path, qr)
