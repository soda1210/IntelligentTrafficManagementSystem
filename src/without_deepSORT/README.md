# Without_deepSORT

## Introduction
This program is designed to analyze traffic impact in a given video file. The input video file must be named "demo.mp4". It will process the video and save the output as "output.mp4". Additionally, the program will generate a text file named "PCU_fps.txt", which represents the traffic impact for each frame of the processed video.

## Download YOLOv4 Weight
- [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

## Usage

### Input
The input video file must be named "demo.mp4" and placed in the same directory as the "yolov4.py" file.

### Output
The processed video file will be saved as output.mp4 in the same directory as the program file. The traffic impact for each frame will be saved as a text file named PCU_fps.txt.

### Requirements
This program requires the following dependencies to be installed:
- Python 3
- OpenCV
- NumPy

## Get start
To run the program, execute the "yolov4.py" file in Python 3.
```bash
  python yolov4.py
```

## References  
- [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
- [yolov4-DNN](https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49)
