# IntelligentTrafficManagementSystem

基於 YOLOv4 物件辨識技術的智慧交通管理系統。該專案旨在透過即時車流辨識，比較兩種不同作法，分別是使用 deepSORT 算法和不使用 deepSORT 算法，來生成即時車流狀況。並且使用 SUMO (Simulation of Urban Mobility) 對交通狀況進行模擬，最後比較辨識即時車流的圖表和使用不同號誌週期圖表的差異。

## feature
使用 YOLOv4 物件辨識技術：專案採用 YOLOv4 物件辨識技術，能夠即時辨識道路上的車輛，提供準確且高效的車流辨識。
比較不同作法：專案比較了使用 deepSORT 算法和不使用 SORT 算法兩種作法的車流辨識結果，並生成對應的車流狀況圖表，以便後續分析和優化交通管理策略。
模擬交通狀況：專案使用 sumo 進行交通狀況模擬，能夠模擬不同號誌週期圖表下的交通流動，並與辨識即時車流的圖表進行對比，提供對交通管理策略的深入分析。

## technology
YOLOv4：進行即時車流辨識的物件辨識技術。
deepSORT 算法：一種在多目標追踪中廣泛使用的算法，用於追踪辨識出的車輛。
sumo (Simulation of Urban Mobility)：一個開放源碼的交通模擬器，用於模擬不同號誌週期圖表下的交通狀況。

## References
- yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
- yolov4_dnn https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49
- yolov4-deepsort https://github.com/nwojke/deep_sort
- SUMO traffic https://www.eclipse.org/sumo/
