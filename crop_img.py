#15 1,2    ./test/02520.jpg
import cv2
     
img = cv2.imread("./test/05405.jpg")
print(img.shape[1])
cropped = img[0:int(img.shape[0]), (int(img.shape[1]/13))*0: (int(img.shape[1]/13))*2]  # 裁剪坐标为[y0:y1, x0:x1]
cv2.imwrite("./test/05405-cro-1.png", cropped)
