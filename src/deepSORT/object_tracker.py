import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# 判斷區間
from interval import Interval

# 608 - 416

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-608',
                    'path to weights file')
# flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_integer('size', 608, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.1, 'iou threshold')
flags.DEFINE_float('score', 0.4, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

def main(_argv):
    # main_lane 車道外框
    # 5min_main_lane
    main_lane = np.array([[950, 600], [850, 1000], [1850, 1000], [1400, 600]], np.int32)
    # mid_lane
    # main_lane = np.array([[500,380], [5, 1080], [950, 1080], [980, 380]], np.int32)

    # 15min main_lane
    # main_lane = np.array([[900, 350], [850, 1000], [1850, 1000], [1350, 400]], np.int32)
    # main_lane = np.array([[850, 350], [850, 1000], [1850, 1000], [1850, 350]], np.int32)

    main_lane_x = Interval(main_lane[1][0], main_lane[2][0])
    main_lane_y = Interval(main_lane[0][1], main_lane[1][1])

    allcount = 0
    carsid = [0]

    # 'car', 'motorbike', 'bus',  'truck'
    car_count = 0
    moto_count = 0
    bus_count = 0
    truck_count = 0

    # 表格
    x_frame = []
    y_traffic = []
    y_area_cars = []
    y_area_pcu = []
    total_fps = []

    traffic_info_save_path = './outputs/size608/tmp.txt'

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0

    file_info = open(traffic_info_save_path, 'w')
    # while video is running
    # 原本是while true
    while cv2.waitKey(1) < 0:
        
        return_value, frame = vid.read()

        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        # allowed_classes = ['person']
        
        allowed_classes = [ 'car', 'motorbike', 'bus',  'truck']
        # allowed_classes = [ 'car', 'motorbike', 'bus']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)


        # 畫車道
        # main_lane = main_lane.reshape((-1, 1, 2))
        # cv2.polylines(frame, [main_lane], True, (0,0,255),3)

        # 計算當前區域車輛總數
        area_car = 0
        area_pcu = 0

        x_frame.append(frame_num)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
            # 計算車流量            
            # pcu => motor=0.75 cars=1 bus,track=3
            
            if (bbox[0]+bbox[2])/2 in main_lane_x and (bbox[1]+bbox[3])/2 in main_lane_y:
                
                area_car += 1

                if class_name == 'car':
                        area_pcu += 1
                elif class_name == 'motorbike':
                        area_pcu += 0.75
                elif class_name == 'bus':
                        area_pcu += 3
                elif class_name == 'truck':
                        area_pcu += 3


                if not track.track_id in carsid:
                    carsid.append(track.track_id)
                    allcount += 1
                    # 'car', 'motorbike', 'bus',  'truck'
                    if class_name == 'car':
                        car_count += 1
                    elif class_name == 'motorbike':
                        moto_count += 1
                    elif class_name == 'bus':
                        bus_count += 1
                    elif class_name == 'truck':
                        truck_count += 1

                #     print(f'新增選中的車輛：{class_name} {track.track_id} , x = {bbox[0]} y = {bbox[1]} \n')
                # else:
                #     print(f'重複選中的車輛：{class_name} {track.track_id} , x = {bbox[0]} y = {bbox[1]} \n')


            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        
        # 計算當前區域車輛總數
        y_traffic.append(allcount)
        y_area_cars.append(area_car)
        y_area_pcu.append(area_pcu)


        file_info.writelines(f'{allcount} {area_car} {area_pcu}\n')
        
        cv2.putText(frame, "total traffic : " + str(allcount), (0, 25), 0, 1, (255,0,0), 2)
        cv2.putText(frame, "total cars : " + str(area_car), (0, 55), 0, 1, (255,0,0), 2)
        cv2.putText(frame, "frame : " + str(frame_num), (0, 85), 0, 1, (0,0,255), 2)
        

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        total_fps.append(fps)

        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    print(f'{(car_count+truck_count*3+bus_count*3+moto_count*0.75)*4}PCU/hr')
    cv2.destroyAllWindows()
    print(f'last frame = {frame_num}')
    # 主車道紅綠燈時向及時長
    # GL = 5s, G = 68s,  GL = 7s, Y = 3s, R = 37s

    x_tick = [0, 4122, 4563, 4714, 6929, 7227, 11322, 11735, 11915, 14135, 14435, 18415]
    x_ticklabel = ['G', 'GL', 'Y', 'R', 'GL', 'G', 'GL', 'Y', 'R', 'GL', 'G', 'GL']
    
    file_info.close()

    file_info = open('./outputs/fps_.txt', 'w')
    for i in range(len(total_fps)):
        if total_fps[i] == None and y_traffic[i] == None:
            file_info.writelines(f'0 0 error\n')
        elif total_fps[i] == None:
            file_info.writelines(f'0  {y_traffic[i]} total_error\n')
        elif y_traffic[i] == None:
            file_info.writelines(f'{total_fps[i]}  0 fps_error\n')
        else:
            file_info.writelines(f'{total_fps[i]}  {y_traffic[i]}\n')
    file_info.close()


    # 當下區域內車輛總數
    plt.title('Vehicles in the area')
    plt.ylabel('cars')
    plt.xlabel('Traffic light')
    plt.xticks(x_tick, x_ticklabel)
    plt.plot(x_frame, y_area_cars)
    plt.show()

    plt.title('PCU in the area')
    plt.ylabel('PCU')
    plt.xlabel('Traffic light')
    plt.xticks(x_tick, x_ticklabel)
    plt.plot(x_frame, y_area_pcu)
    plt.show()

    # 累積車輛總數
    plt.title('Total traffic')
    plt.ylabel('cars')
    plt.xlabel('Traffic light')
    plt.xticks(x_tick, x_ticklabel)
    plt.plot(x_frame, y_traffic)
    plt.show()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
