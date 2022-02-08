import cv2
import numpy as np
import random
import sys
import os

# check input file argument exists
if (len(sys.argv) != 2):
    sys.exit("needs one input file")

# create input file path (check it exists?)
inputFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)),sys.argv[1])

inputImage = cv2.imread(inputFilePath, 1)
# inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB) does it need to be rgb? we never see it

imageHeight = inputImage.shape[0]
imageWidth = inputImage.shape[1]

brushImage = cv2.imread('brushes/1.jpg', 0)

brushHeight = brushImage.shape[0]
brushWidth = brushImage.shape[1]

# print(brushHeight)

brushTopBorder = imageHeight - brushHeight
brushRightBorder = imageWidth - brushWidth

def generateTransform(x, y):

    tx = ((random.random() * (600 )) - 100 ) * -1
    ty = ((random.random() * (500 )) - 100 )

    ty = (y - 150)
    tx = (x - 150)

    M = np.float32([[1,0,tx],[0,1,ty]])
    return M

blankImage = np.ones((imageHeight,imageWidth,3))

baseImage = inputImage.copy()
baseImage = np.zeros((imageHeight,imageWidth,3))

brush = brushImage.astype(float) / 255

for x in range(100):
    randX = int(random.random() * imageWidth)
    randY = int(random.random() * imageHeight)

    # print('x: ' + str(randX) + ' y: ' + str(randY))

    # print(inputImage[randY][randX])

    border = cv2.copyMakeBorder(
            brush,
            top=0,
            bottom=brushTopBorder,
            left=0,
            right=brushRightBorder,
            borderType=cv2.BORDER_CONSTANT,
            value=[0]
                )
        
    sampleBrushImage = cv2.warpAffine(border,generateTransform(randX, randY),(imageWidth,imageHeight))

    sampleBrushImage = np.tile(sampleBrushImage[:, :, None], [1, 1, 3])

    targetColor = blankImage.copy()
    targetColor[:,:] = inputImage[randY][randX].astype(int)
    targetColor = cv2.convertScaleAbs(targetColor)

    # print(targetColor.shape)
    # print(targetColor[randY][randX])

    # print(blankImage.dtype)
    # print(sampleBrushImage.dtype)
    # print(targetColor.dtype)

    sampleImage = cv2.add( cv2.multiply(baseImage.astype(float), cv2.subtract(blankImage.astype(float), sampleBrushImage.astype(float)) ), cv2.multiply(targetColor.astype(float), sampleBrushImage.astype(float)) )

    baseImage = cv2.convertScaleAbs(sampleImage)

baseImage = np.concatenate((inputImage,baseImage), axis=1)

cv2.imshow('image', baseImage)
cv2.waitKey(0)
cv2.imwrite('output.jpg',baseImage)