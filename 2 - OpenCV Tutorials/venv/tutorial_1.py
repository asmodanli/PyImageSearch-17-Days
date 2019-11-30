import imutils
import cv2

image = cv2.imread("C:/Users/Asmod/Desktop/Proje/2 - OpenCV Tutorials/venv/image.jpg")
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))
cv2.imshow("Image", image)
cv2.waitKey(0)

# depth -> channel sayısı. bgr kanalında çalıştığımız için derinlik 3

(B, G, R) = image[100, 50]
print("R = {}, G = {}, b = {}".format(R, G, B))  # X = 50, Y = 100 deki RGB pixelleri

roi = image[144:308, 170:280]  # yüz koordinatı / [startY:endY, startX:endX
cv2.imshow("roi", roi)
cv2.waitKey(0)

# Manual Resize
ratio = 200.0 / w
dim = (300, int(h * ratio))
resized = cv2.resize(image, dim)
cv2.imshow("resize", resized)
cv2.waitKey(0)

# Auto Resize
resized = imutils.resize(image, width=200)
cv2.imshow("imutils resize", resized)
cv2.waitKey(0)

# Rotate image, kenarlar kırpılır
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)  # M -> rotation matrix, -45 -> saay yönünde 45 derece
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("OpenCv Rotation", rotated)
cv2.waitKey(0)

# Rotate Image, imutils
rotated = imutils.rotate_bound(image, 45)
cv2.imshow("imutils Rotation", rotated)
cv2.waitKey(0)

# Smooth image
blurred = cv2.GaussianBlur(image, (11, 11), 0)  # 11,11 kernel
blurred2 = cv2.GaussianBlur(image, (11, 11), 15)  # 11,11 kernel
cv2.imshow("Blurred", blurred)
cv2.imshow("Blurred2", blurred2)
cv2.waitKey(0)

# Draw rectangle
output = image.copy()
cv2.rectangle(output, (170, 144), (280, 308), (0, 0, 255), 2)  # 2 -> kalınlık
cv2.imshow("Rectangle", output)
cv2.waitKey(0)

# Draw Circle
output = image.copy()
cv2.circle(output, (300, 150), 20, (255, 0, 0), -1)  # 20 -> yarıçap pikseli
cv2.imshow("Circle", output)
cv2.waitKey(0)

# Draw Line
output = image.copy()
cv2.line(output, (0, 0), (300, 300), (0, 255, 0), 5)
cv2.imshow("Line", output)
cv2.waitKey(0)

# Put Text
output = image.copy()
cv2.putText(outputz, "Hello World!", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Put Text", output)
cv2.waitKey(0)
