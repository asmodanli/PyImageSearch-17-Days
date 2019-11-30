import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Image", image)
cv2.waitKey(0)

# convert grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

# Edge detection
edged = cv2.Canny(gray, 30, 150)  # 30 -> min treshold (minVal), 150 -> max treshold (minVal)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

edged2 = cv2.Canny(gray, 45, 180)  # 30 -> min treshold (minVal), 150 -> max treshold (minVal)
cv2.imshow("Edged2", edged2)
cv2.waitKey(0)

# Thresholding -> İkili görüntüye (binary) çevirir -> yani siyah beyaz
# 225-255 den küçük sayıları 225-255 den büyük yaptık
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thres", thresh)
cv2.waitKey(0)
"""
# Detect and Draw Contour
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # beyaz pixeller sayesinde sontour buluyor
cnts = imutils.grab_contours(cnts)
output = image.copy()

for i in cnts:
    cv2.drawContours(output, [i], -1, (240, 0, 159), 3)  # 3 px'li, mor renkli ((240, 0, 159)), içe doğru kalınlaşan (-1) çizgiyle contourları çiz
    cv2.imshow("Contours", output)
    cv2.waitKey(0)

# Draw total number of the contours
text = "I found {} objects".format(len(cnts))
cv2.putText(output, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)
"""
# Erosion and dilation
# Binary resimlerde gürültüyü azaltmak için kullanılır.

mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=1)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)
# iterations bu işlemin kaç kere yapıldığı anlamına geliyor. Ne kadar çoksa objeler o kadar incelir
mask2 = thresh.copy()
mask2 = cv2.erode(mask2, None, iterations=10)
cv2.imshow("Eroded2", mask2)
cv2.waitKey(0)

# Dilation objeleri büyütür
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Dilated", mask)
cv2.waitKey(0)

# FARKLI BİR SAYIM
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # beyaz pixeller sayesinde sontour buluyor
cnts = imutils.grab_contours(cnts)
output = image.copy()

for i in cnts:
    cv2.drawContours(output, [i], -1, (240, 0, 159), 3)  # 3 px'li, mor renkli ((240, 0, 159)), içe doğru kalınlaşan (-1) çizgiyle contourları çiz
    cv2.imshow("Contours", output)
    cv2.waitKey(0)

# Draw total number of the contours
text = "I found {} objects".format(len(cnts))
cv2.putText(output, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)


# python  tutorial_2.py --image tetris.png

