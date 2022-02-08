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
brushSmallImage = cv2.imread('brushes/2.png', 0)
brushSmallerImage = cv2.imread('brushes/3.png', 0)

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

def generateRandomRotation(rows, cols):
    rotation = random.random() * 360
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),rotation,1)
    return M

def calcDifference(inImg, alteredImg):
    diff1 = cv2.subtract(inImg, alteredImg, dtype=cv2.CV_64F) #values are too low
    totalDiff = np.absolute(diff1)
    totalDiff = np.sum(totalDiff)
    return totalDiff

blankImage = np.ones((imageHeight,imageWidth,3))

baseImage = inputImage.copy()
baseImage = np.zeros((imageHeight,imageWidth,3))

brush = brushImage.astype(float) / 255
brushSmall = brushSmallImage.astype(float) / 255
brushSmaller = brushSmallerImage.astype(float) / 255

for x in range(400):
    randX = int(random.random() * imageWidth)
    randY = int(random.random() * imageHeight)

    # print('x: ' + str(randX) + ' y: ' + str(randY))

    # print(inputImage[randY][randX])

    inBrush = brush

    if(x>400):
        inBrush = brushSmall
    if(x>800):
        inBrush = brushSmaller

    rotatedBrush = cv2.warpAffine(inBrush,generateRandomRotation(300, 300),(brush.shape[1],brush.shape[0]))

    border = cv2.copyMakeBorder(
            rotatedBrush,
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

    difference = calcDifference(inputImage, sampleImage)
    diffBase = calcDifference(inputImage, baseImage)

    if (difference <= diffBase):
        baseImage = cv2.convertScaleAbs(sampleImage)
    

comparisonImage =  np.absolute(cv2.subtract(inputImage, baseImage, dtype=cv2.CV_64F).astype(int))

print('stage 1')
# print(comparisonImage[:,:])

comparisonImage = cv2.cvtColor(np.uint8(comparisonImage), cv2.COLOR_BGR2GRAY)
comparisonImage = cv2.blur(comparisonImage, (30,30))

for x in range(800):

    maskValue = 0
    while(maskValue < 20):
        randX = int(random.random() * imageWidth)
        randY = int(random.random() * imageHeight)

        maskValue = comparisonImage[randY,randX]
        # print(maskValue)
    # print('Using: ' + str(maskValue))

    # print('x: ' + str(randX) + ' y: ' + str(randY))

    # print(inputImage[randY][randX])

    inBrush = brushSmall

    if(x>300):
        inBrush = brushSmaller

    rotatedBrush = cv2.warpAffine(inBrush,generateRandomRotation(300, 300),(brush.shape[1],brush.shape[0]))

    border = cv2.copyMakeBorder(
            rotatedBrush,
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

    difference = calcDifference(inputImage, sampleImage)
    diffBase = calcDifference(inputImage, baseImage)

    if (difference <= diffBase):
        baseImage = cv2.convertScaleAbs(sampleImage)

print('stage 2')

comparisonImage =  np.absolute(cv2.subtract(inputImage, baseImage, dtype=cv2.CV_64F).astype(int))


comparisonImage = cv2.cvtColor(np.uint8(comparisonImage), cv2.COLOR_BGR2GRAY)
comparisonImage = cv2.blur(comparisonImage, (10,10))

for x in range(500):

    maskValue = 0
    while(maskValue < 50):
        randX = int(random.random() * imageWidth)
        randY = int(random.random() * imageHeight)

        maskValue = comparisonImage[randY,randX]
        # print(maskValue)
    # print('Using: ' + str(maskValue))

    # print('x: ' + str(randX) + ' y: ' + str(randY))

    # print(inputImage[randY][randX])

    inBrush = brushSmall

    if(x>1):
        inBrush = brushSmaller

    rotatedBrush = cv2.warpAffine(inBrush,generateRandomRotation(300, 300),(brush.shape[1],brush.shape[0]))

    border = cv2.copyMakeBorder(
            rotatedBrush,
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

    difference = calcDifference(inputImage, sampleImage)
    diffBase = calcDifference(inputImage, baseImage)

    if (difference <= diffBase):
        baseImage = cv2.convertScaleAbs(sampleImage)

print('stage 3')

for j in range(imageWidth):
    for i in range(imageHeight):
        # print(comparisonImage[i,j])
        if(comparisonImage[i,j] > 20):
            comparisonImage[i,j] = 255

comparisonImage = np.tile(comparisonImage[:, :, None], [1, 1, 3])

baseImage = np.concatenate((inputImage,baseImage,comparisonImage), axis=1)

cv2.imshow('image', baseImage)
cv2.waitKey(0)
cv2.imwrite('output.jpg',baseImage)