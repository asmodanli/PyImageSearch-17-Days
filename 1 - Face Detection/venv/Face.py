import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input page")
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# --image = path to the input image,
# --prototxt = path to caffe prototxt file
# --model = path to pretrained caffe model

print("[INFO] loading model")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"]) #load model from disk
# --prototxt ve --model ile modeli kullanıyoruz. Modeli "net" de saklıyoruz

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
# blobFromImage -> Görüntülerin okunarak sinir ağı girişine hazırlar (blob haline getirir)

print("[INFO] computinf object deetections...")
net.setInput(blob) # pass blob through the network and obtain the detections and predictions
detections = net.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2] #güven

    if confidence > args["confidence"]: #güven eşiğinden (0.5) büyük mü?
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int") #etrafına çizilecek karenin ölçüleri

        text = "{: .2f}%".format(confidence * 100) #algılama olasılığı
        y = startY - 10 if startY - 10 > 10 else startY + 10 #yüz çok yukarıdaysa 10 pixel aşağı indir
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2) #kareyi çiz

cv2.imshow("Output", image)
cv2.waitKey(0)


