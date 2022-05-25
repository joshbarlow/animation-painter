import cv2
import numpy as np
import random
import sys
import os

def animPaintBatch():
    # check input file argument exists
    if (len(sys.argv) != 2):
        sys.exit("needs one input file")

    # create input file path (check it exists?)
    inputFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)),sys.argv[1])

    inputFiles = sorted(os.listdir(inputFilePath))

    # for file in inputFiles:
    #     print(file)

    brushImage = cv2.imread('brushes/1.jpg', 0)
    brushSmallImage = cv2.imread('brushes/2.png', 0)
    brushSmallerImage = cv2.imread('brushes/3.png', 0)

    outFileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'output')
    os.chdir(outFileDir)

    for file in inputFiles:
        if('.png' in file):
            inFilePath = os.path.join(inputFilePath,file)
            outFilePath = file
            animPaint(inFilePath,outFilePath,brushImage,brushSmallImage,brushSmallerImage)
            print(inFilePath + ' -> ' + outFilePath)

def animPaint(inFile, outFile,brushImage,brushSmallImage,brushSmallerImage):
    # check input file argument exists
    # if (len(sys.argv) != 2):
    #     sys.exit("needs one input file")

    # create input file path (check it exists?)
    # inputFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)),sys.argv[1])

    inputImage = cv2.imread(inFile, 1)
    # inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB) does it need to be rgb? we never see it

    imageHeight = inputImage.shape[0]
    imageWidth = inputImage.shape[1]

    # brushImage = cv2.imread('brushes/1.jpg', 0)
    # brushSmallImage = cv2.imread('brushes/2.png', 0)
    # brushSmallerImage = cv2.imread('brushes/3.png', 0)

    brushHeight = brushImage.shape[0]
    brushWidth = brushImage.shape[1]

    # print(brushHeight)

    brushTopBorder = imageHeight - brushHeight
    brushRightBorder = imageWidth - brushWidth

    baseImage = np.zeros((imageHeight,imageWidth,3))

    maskImage = np.ones((imageHeight,imageWidth,1)) * 100

    # First Pass 400
    outImage = paintIteration(baseImage, inputImage, brushImage, brushTopBorder, brushRightBorder, maskImage, 400)
    print('stage 1')

    comparisonImage =  np.absolute(cv2.subtract(inputImage, outImage, dtype=cv2.CV_64F).astype(int))
    comparisonImage = cv2.cvtColor(np.uint8(comparisonImage), cv2.COLOR_BGR2GRAY)
    comparisonImage = cv2.blur(comparisonImage, (30,30))

    # Second Pass 400
    outImage = paintIteration(outImage, inputImage, brushSmallImage, brushTopBorder, brushRightBorder, comparisonImage, 400)
    print('stage 2')

    comparisonImage =  np.absolute(cv2.subtract(inputImage, outImage, dtype=cv2.CV_64F).astype(int))
    comparisonImage = cv2.cvtColor(np.uint8(comparisonImage), cv2.COLOR_BGR2GRAY)
    comparisonImage = cv2.blur(comparisonImage, (10,10))

    # Third Pass 500
    outImage = paintIteration(outImage, inputImage, brushSmallerImage, brushTopBorder, brushRightBorder, comparisonImage, 500)
    print('stage 3')

    cv2.imshow('image', outImage)
    cv2.waitKey(0)

    cv2.imwrite(outFile,outImage)


def generateTransform(x, y):
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

def paintIteration(baseImageIn, targetImage, brushIn, brushTopBorder, brushRightBorder, maskIn, iterations):
    
    imageHeight = baseImageIn.shape[0]
    imageWidth = baseImageIn.shape[1]

    blankImage = np.ones((imageHeight,imageWidth,3))

    baseImage = baseImageIn.copy()

    brush = brushIn.astype(float) / 255

    mask = maskIn.copy()

    for x in range(iterations):

        maskValue = 0
        while(maskValue < 20):
            randX = int(random.random() * imageWidth)
            randY = int(random.random() * imageHeight)

            maskValue = mask[randY,randX]

        inBrush = brush

        rotatedBrush = cv2.warpAffine(brush,generateRandomRotation(300, 300),(brush.shape[1],brush.shape[0]))

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
        targetColor[:,:] = targetImage[randY][randX].astype(int)
        targetColor = cv2.convertScaleAbs(targetColor)

        # print(targetColor.shape)
        # print(targetColor[randY][randX])

        # print(blankImage.dtype)
        # print(sampleBrushImage.dtype)
        # print(targetColor.dtype)

        sampleImage = cv2.add( cv2.multiply(baseImage.astype(float), cv2.subtract(blankImage.astype(float), sampleBrushImage.astype(float)) ), cv2.multiply(targetColor.astype(float), sampleBrushImage.astype(float)) )

        difference = calcDifference(targetImage, sampleImage)
        diffBase = calcDifference(targetImage, baseImage)

        if (difference <= diffBase):
            baseImage = cv2.convertScaleAbs(sampleImage)

    return baseImage

animPaintBatch()