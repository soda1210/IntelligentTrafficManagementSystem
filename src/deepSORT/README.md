# YOLOv4-deepSORT

## Environments
```bash
  conda env create -f conda-cpu.yml
  conda activate yolov4-gpu
  
  pip install -r requirements-gpu.txt
```

## Download YOLOv4 Weight
- [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)

## Get Started
```bash
  python save_model.py --modal yolov4
  python object_tracker.py --video <your path> --output ./outputs/demo.avi --model yolov4
```

## References  
- [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
- [yolov4-deepSORT](https://github.com/theAIGuysCode/yolov4-deepsort.git)
