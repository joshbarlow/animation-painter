# animation-painter
A brute-force animated painting experiment. The script creates hundreds of random brush strokes and only keeps the ones which decrease the difference in pixel values between the painting and the target image.

https://user-images.githubusercontent.com/7104517/170365426-0e017c29-a35c-47f5-ab65-5a37a54e6e45.mp4

### Usage:
```
python3 animPaint.py /path/to/folder/containing/images
```

In my tests I tried to have images that were roughly 960x480, but any image size will work.

Built with python 3.8.7 and the following libraries:
- numpy 1.22.1
- openCV 4.5.5.62
