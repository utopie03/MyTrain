import cv2

img = cv2.imread("lenna.jpg",0)
cv2.imshow("img",img)
cv2.imwrite("lenna01.jpg",img)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(img.shape)