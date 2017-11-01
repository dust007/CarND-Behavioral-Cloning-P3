import cv2

image = cv2.imread("./examples/center.jpg")
image1 = cv2.flip(image, 1)
cv2.imwrite('center_flip.png',image1)
