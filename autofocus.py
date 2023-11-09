import cv2 as cv
img = cv.imread('autofocus\\cactus1.jpg')

height = img.shape[0]
width = img.shape[1]

print('height = ', height, 'width = ', width, '\n')

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#cv.imshow("Display window", img)
#k = cv.waitKey(0)

def sum_modified_laplacian_af(img):
    FM_SML = 0
    imgShapeX = img.shape[0] - 2
    imgShapeY = img.shape[1] - 2
    for i in range(1, imgShapeX):
        for j in range(1, imgShapeY):
            FM_SML += abs(-1*img[i-1, j] + 2*img[i, j] - img[i+1, j]) + abs(-1*img[i, j-1] + 2*img[i, j] - img[i, j+1])
    print("SML = ", FM_SML, '\n')
    return FM_SML

af = sum_modified_laplacian_af(img_gray)