from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import cv2
import argparse
import imutils

# Find Edges
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="poth to the input image")
args = vars(ap.parse_args())

answer_key = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert grayscale
blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # gürültüyü azaltmak için blurred
edged = cv2.Canny(blurred, 75, 200)  # edge detector

cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find Contours
# CHAIN_APPROX_SIMPLE -> contouru sıkıştırır, hafıza tasarrufu yapar
# RETR_EXTERNAL -> parent contouru döndürür
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # cnts'deki contourleri büyükten küçüğe sıralıyor

    for i in cnts:
        peri = cv2.arcLength(i, True)  # çevre uzunluğu, True -> kapalı contour
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # contouru belirlenen hassasiyete bağlı olarak  başka bir şekle
        if len(approx) == 4:  # benzetir ikinci parametre -> istenen contourle yaklaşık
            docCnt = approx  # contour arsındaki fark (hassasiyet)
            break

""" cv2.drawContours(image, [docCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Contours", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() """

# kuşbakışı görüntü aldık
paper = four_point_transform(image, docCnt.reshape(4, 2))  # four_point_transform -> contourları belirli bir şekilde
warped = four_point_transform(gray, docCnt.reshape(4, 2))  # sıralar. Bölgeye bir perspektif dönüşümü sağlar.

# Thresholding
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # Karalı alanları bulduk

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questions = []

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)  # dikdörtgen hesaplar
    ar = w / float(h)  # en - boy oranı

    if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:  # işaretlenmiş mi anlamak için en boy aralığı vermiş
        questions.append(c)

questions = contours.sort_contours(questions, method="top-to-bottom")[0]
correct = 0

for (q, i) in enumerate(np.arange(0, len(questions), 5)):  # her sorunun 5 şıkkı var
    cnts = contours.sort_contours(questions[i: i + 5])[0]
    bubled = None

    for (j, c) in enumerate(cnts):
        mask = np.zeros(thresh.shape, dtype="uint8")  # sadece o anki yuvarlağı ortaya çıkartacak bir mask
        cv2.drawContours(mask, [c], -1, 255, -1)

        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        if bubled is None or total > bubled[0]:
            bubled = (total, j)

    color = (0, 0, 255)
    k = answer_key[q]

    if k == bubled[1]:  # doğru mu
        color = (0, 255, 0)
        correct += 1

    cv2.drawContours(paper, [cnts[k]], -1, color, 3)

score = (correct / 5.0) * 100
print("[Info] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
cv2.imshow("original", image)
cv2.imshow("result", paper)
cv2.waitKey(0)