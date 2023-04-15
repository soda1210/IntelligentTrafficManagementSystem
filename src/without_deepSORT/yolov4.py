import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

# 確認物件的閥值
CONFIDENCE_THRESHOLD = 0.2
# 重複的BBox機率閥值
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

save_path = './416PCU.txt'

# 左轉道區域  左上[y,x] 左下 右下 右上
left_lane = np.array([[950, 530], [850, 1000], [1150, 1000], [1100, 530]], np.int32)
main_lane = np.array([[950, 600], [850, 1000], [1850, 1000], [1400, 600]], np.int32)

# real_left_lane_area = np.array([[950, 530], [950, 1000], [1100, 1000], [1100, 530]], np.int32)

# 各項車輛統計
car_count = 0
moto_count = 0
bus_count = 0
truck_count = 0

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

vc = cv2.VideoCapture("demo.mp4")

# 製作影片
width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vc.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', codec, 3, (width, height))

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
# model.setInputParams(size=(512, 512), scale=1/255, swapRB=True)
# model.setInputParams(size=(608, 608), scale=1/255, swapRB=True)

# 交通影響量
tcount = np.array(10 * [0])
j = 0
pcu = []
area_cars = []
now_frame = 0

f = open(save_path, 'w')
while 1:
    now_frame += 1
    
    #  減少幀數 接近real-time
    for i in range(0, 20):
        (grabbed, frame) = vc.read()

    if not grabbed:
        break

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    start_drawing = time.time()
    
    #  左轉車道及主車道車輛數
    turnleft = 0
    main_cars = 0
    main_moto = 0
    tmp = 0

    for (classid, score, box) in zip(classes, scores, boxes):
        # classid  => 2 = car, 3 = motorbike, 5 = bus, 7 = truck
        if classid == 2 or classid == 3 or classid == 5 or classid == 7:
            # print(f"class = {classid},  score = {score}, box = {box}")
            
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_names[classid], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # [[950, 600], [850, 1000], [1850, 1000], [1400, 600]]
            
            # 主車道車輛計數
            if box[0]+box[2]/2 >= 850 and box[0]+box[2]/2 <= 1850 and box[1]+box[3]/2 >= 600 and box[1]+box[3]/2 <= 1000:
                # 汽車
                if classid == 2:
                    car_count += 1
                    main_cars += 1
                    tmp += 1
                # 機車
                if classid == 3:
                    moto_count += 1
                    main_moto += 1
                    tmp += 0.8
                # 公車及貨車
                if classid >= 5:
                    bus_count += 1
                    truck_count += 1
                    main_cars += 1
                    tmp += 2

            # 左轉車輛計數
            if box[0]+box[2]/2 >= 950 and box[0]+box[2]/2 <= 1100 and box[1]+box[3]/2 >= 530 and box[1]+box[3]/2 <= 1000:
                turnleft += 1
                # print("turnleft", classid, box)

    tcount[j] = tmp
    j+= 1
    if j == 10:
        j = 0
    
    main_lane = main_lane.reshape((-1, 1, 2))
    cv2.polylines(frame, [main_lane], True, (0,0,255),3) 
    cv2.polylines(frame, [left_lane], True, (255,0,0),3)
    end_drawing = time.time()

    cv2.rectangle((frame), (0, 0), (580, 150), (255, 255, 255), -1)

    fps_label = "FPS: %.2f (drawing time %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, "PCU/10fps : " + str(np.sum(tcount)), (0, 55), 0, 1, (255,0,0), 2)
    cv2.putText(frame, "frame : " + str(now_frame), (0, 85), 0, 1, (0, 0, 255), 2)
    cv2.putText(frame, "waiting turn left : " + str(turnleft), (0, 115), 0, 1, (0, 165, 255), 2)
    cv2.putText(frame, "Main cars : " + str(main_cars)+" Motorcycle : " + str(main_moto), (0, 145), 0, 1, (139, 0, 0), 2)
    out.write(frame)
    cv2.imshow("detections", frame)

    print("交通影響量：", np.sum(tcount), "\nframe：", now_frame)
    pcu.append(np.sum(tcount))

    print("區域內的車輛數：", main_cars , "機車數：",main_moto)
    area_cars.append(main_cars + main_moto)
    

    f.writelines(f'{now_frame} {np.sum(tcount)}\n')
    
    # cv2.imwrite("./img/output.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    # cv2.waitKey(0)
    if cv2.waitKey(1) == ord('q'):
        now_frame += 1
        break

f.close()

out.release()
now_frame = range(now_frame-1)
cv2.destroyAllWindows()
plt.title('PCU/10fps')
plt.ylabel('PCU')
plt.xlabel('Traffic light')
x_tick = [0, 206, 227, 236, 347, 362, 566, 587, 596, 707, 722, 926]
x_ticklabel = ['G', 'GL', 'Y', 'R', 'GL', 'G', 'GL', 'Y', 'R', 'GL', 'G', 'GL']
plt.xticks(x_tick, x_ticklabel)
plt.plot(now_frame, pcu)
plt.show()

plt.title('Area cars')
plt.ylabel('Cars')
plt.xlabel('Traffic light')
x_tick = [0, 206, 227, 236, 347, 362, 566, 587, 596, 707, 722, 926]
x_ticklabel = ['G', 'GL', 'Y', 'R', 'GL', 'G', 'GL', 'Y', 'R', 'GL', 'G', 'GL']
plt.xticks(x_tick, x_ticklabel)
plt.plot(now_frame, area_cars)
plt.show()