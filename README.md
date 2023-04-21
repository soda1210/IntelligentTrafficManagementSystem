# IntelligentTrafficManagementSystem

An adaptive traffic signal control mechanism based on YOLOv4 object recognition technology to reduce traffic congestion. The project aims to generate real-time traffic flow by comparing two methods, the first one using deepSORT algorithm and the other one is our propose algorithm by real-time traffic flow recognition. In addition, the traffic flow is simulated using SUMO (Simulation of Urban Mobility) to change the traffic signals by both methods. Finally, we compare the effects of both methods on traffic flow and  recognition speed and the experimental results prove that our method is better than the another one.

## Features
Using YOLOv4 Object Detection Technology: The project utilizes YOLOv4 object detection technology to provide accurate and efficient vehicle detection on the road, enabling precise traffic flow recognition.

Comparing Different Approaches: The project compares the results of traffic flow recognition using two approaches: with and without the use of the deepSORT algorithm. Traffic flow status charts are generated for subsequent analysis and optimization of traffic management strategies.

Simulating Traffic Conditions: This project utilizes SUMO (Simulation of Urban Mobility) to simulate adaptive traffic signal control strategies and compare them with traditional fixed signal control strategies. This provides a analysis of traffic management strategies.

## Technologies
YOLOv4: Object detection algorithm that can quickly and accurately recognize objects in images.

deepSORT: A target tracking algorithm that can identify different objects and continuously track their positions.

SUMO (Simulation of Urban Mobility): Software that simulates city traffic flow, which helps researchers to simulate various traffic scenarios and conduct traffic flow analysis, optimization, and other research.

## References
- [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
- [yolov4_dnn](https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49)
- [yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort.git)
- [SUMO traffic](https://www.eclipse.org/sumo/)
