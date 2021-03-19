import cv2
import uav
import time
import wifi_init
import paddlehub as hub
import os
from multiprocessing import Process, Queue, Manager
import numpy as np
import math
import pid
import deepsort
import copy
import av
import FPS


def object_detector(frame, obj_res):
    """物体检测进程"""
    # 加载模型，可选模型 ssd_mobilenet_v1_pascal 和 yolov3_mobilenet_v1_coco2017
    detector = hub.Module(name="yolov3_mobilenet_v1_coco2017")
    while len(frame) == 0:
        time.sleep(0.1)
    while True:
        res = detector.object_detection(
            images=[frame[-1]],
            batch_size=1,
            use_gpu=True,
            score_thresh=0.4,
            visualization=False
        )
        obj_res.append(res)
        if len(obj_res) > 1:
            obj_res.pop(0)


def face_detector(frame, face_res):
    """人脸检测进程"""
    face_detector = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_640")
    while len(frame) == 0:
        time.sleep(0.1)
    while True:
        res = face_detector.face_detection(
            images=[frame[-1]],
            use_gpu=False,
            visualization=False,
            confs_threshold=0.8)
        face_res.append(res)
        if len(face_res) > 1:
            face_res.pop(0)


def human_track(frame, target_res):
    """目标跟踪进程"""
    human_track = deepsort.DeepSort(use_gpu=True, threshold=0.2)
    while len(frame) == 0:
        time.sleep(0.1)
    while 1:
        res = human_track.update(frame[-1])
        if res is not None:
            target_res.append(res)
        else:
            target_res.append([])
        if len(target_res) > 1:
            target_res.pop(0)


def pose_estimation(frame, pose_res):
    """骨骼关键点检测进程"""
    # 可选模型 openpose_body_estimation 和 openpose_hands_estimation
    pose_estimation = hub.Module(name="openpose_body_estimation")
    while len(frame) == 0:
        time.sleep(0.1)
    while True:
        res = pose_estimation.predict(
            img=frame[-1],
            # scale=[0.5, 1.0, 1.5, 2.0],  # 识别手部关键点时,使用图片的不同尺度
            visualization=False)
        pose_res.append(res)
        if len(pose_res) > 1:
            pose_res.pop(0)


def depth_estimation(frame, depth_res):
    """深度估计进程"""
    model = hub.Module(name='MiDaS_Small', use_gpu=True)
    while len(frame) == 0:
        time.sleep(0.1)
    while True:
        outputs = model.depth_estimation(
            images=[frame[-1]],
            paths=None,
            batch_size=1)
        res = outputs[0]
        depth_res.append(res)
        if len(depth_res) > 1:
            depth_res.pop(0)


def quaternion2yaw(q0, q1, q2, q3):
    """四元数转偏航角"""
    degree = math.pi / 180
    siny = 2.0 * (q0 * q3 + q1 * q2)
    cosy = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    return int(math.atan2(siny, cosy)/degree)


def drawPoints(img, points):
    """绘制航迹"""
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 0, 255), cv2.FILLED)
    cv2.putText(img, f"({((points[-1][0] - 500) / 10):.1f}, {-((points[-1][1] - 500) / 10):.1f})m",
                (int(points[-1][0]) + 10, int(points[-1][1]) + 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)


def drawArrows(img, point, yaw):
    """绘制箭头"""
    length = 8
    angle = 45
    yaw += 90
    fpoint = [int(point[0] + length * math.cos(math.radians(yaw - 180))),
              int(point[1] + length * math.sin(math.radians(yaw - 180)))]
    bpoint = [int(point[0] + length * math.cos(math.radians(yaw)) / 4),
              int(point[1] + length * math.sin(math.radians(yaw)) / 4)]
    lpoint = [int(point[0] + length * math.cos(math.radians(yaw - angle))),
              int(point[1] + length * math.sin(math.radians(yaw - angle)))]
    rpoint = [int(point[0] + length * math.cos(math.radians(yaw + angle))),
              int(point[1] + length * math.sin(math.radians(yaw + angle)))]
    area = np.array([fpoint, lpoint, bpoint, rpoint])
    cv2.fillConvexPoly(img, area, (0, 255, 0))


pos_x, pos_y = 500, 500  # 轨迹起始点坐标
points = [[500, 500]]
yaw = 0  # 偏航角
battery = 0  # 电量
wifi_strength = 0  # 信号强度
mouse_x, mouse_y = 0, 0
target_index = 0


def capture_mouse_event(event, x, y, flags, params):
    global mouse_x, mouse_y, target_index
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y
    elif event == cv2.EVENT_RBUTTONDOWN:
        target_index = 0


def handler(event, sender, data, **args):
    drone = sender
    global pos_x, pos_y, yaw, battery, wifi_strength, points
    if event is drone.EVENT_LOG_DATA:  # 每秒15帧数据
        pos_x += data.mvo.vel_y / 15
        pos_y -= data.mvo.vel_x / 15
        """记录飞行轨迹点"""
        if (points[-1][0] != pos_x) or (points[-1][1] != pos_y):
            points.append([pos_x, pos_y])
        yaw = quaternion2yaw(data.imu.q0, data.imu.q1, data.imu.q2, data.imu.q3)
    if event is drone.EVENT_FLIGHT_DATA:
        battery = data.battery_percentage
        wifi_strength = data.wifi_strength


def data_processing(frame, obj_res, face_res, target_res, pose_res, depth_res):
    """数据综合处理进程"""
    WIFI = wifi_init.wifi()  # 实例化wifi类
    wifi_init.connect(WIFI)

    global pos_x, pos_y, yaw, battery, wifi_strength, points, mouse_x, mouse_y, target_index
    fly = uav.Uav()
    drone = fly.drone
    drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
    drone.subscribe(drone.EVENT_LOG_DATA, handler)
    cv2.namedWindow("UAV")
    cv2.setMouseCallback("UAV", capture_mouse_event)

    retry = 3
    container = None
    while container is None and 0 < retry:
        retry -= 1
        try:
            container = av.open(drone.get_video_stream())
        except av.AVError as ave:
            print(ave)
            print('retry...')

    # skip first 300 frames
    frame_skip = 300
    fps = FPS.FPS()

    width, height = 960, 720  # 图像宽度、高度
    command = [0] * 4
    Roll_pid = pid.PID(0.002, 0, 0.0001)
    x_err_time = 0

    while True:
        for raw_frame in container.decode(video=0):
            if 0 < frame_skip:
                frame_skip = frame_skip - 1
                continue
            start_time = time.perf_counter()
            """图像获取"""
            raw_image = cv2.cvtColor(np.array(raw_frame.to_image()), cv2.COLOR_RGB2BGR)
            frame.append(raw_image)
            if len(frame) > 1:
                frame.pop(0)
            image = copy.deepcopy(raw_image)
            """物体检测数据处理"""
            if len(obj_res) > 0:
                for data in obj_res[-1][0]["data"]:
                    if data['label'] != "person":
                        left = int(data["left"])
                        top = int(data["top"])
                        right = int(data["right"])
                        bottom = int(data["bottom"])
                        cv2.rectangle(image, (left, top), (right, bottom), (255, 100, 100), 2)
                        cv2.putText(image, f"{data['label']}", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            """人脸识别数据处理"""
            if len(face_res) > 0:
                faceListC = []
                faceListSize = []
                for face in face_res[-1][0]["data"]:
                    left = int(face["left"])
                    top = int(face["top"])
                    right = int(face["right"])
                    bottom = int(face["bottom"])
                    faceListC.append([(right + left) // 2, (bottom + top) // 2])
                    faceListSize.append(right + bottom - left - top)
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)  # 在图片中标注人脸，并显示
                """人脸跟踪，输出偏航控制数据"""
                # if len(faceListSize) != 0:
                #     i = faceListSize.index(max(faceListSize))
                #     x_err = faceListC[i][0] - width / 2
                #     command[1] = Roll_pid.update(x_err)
                #     x_err_time = time.perf_counter()
            """目标跟踪数据处理"""
            if len(target_res) > 0:
                for index, output in enumerate(target_res[-1]):
                    if mouse_x != 0:
                        if output[0] < mouse_x < output[2] and output[1] < mouse_y < output[3]:
                            target_index = output[-1]
                            mouse_x = 0
                    if target_index != 0 and target_index == output[-1]:
                        cv2.rectangle(image, (output[0], output[1]), (output[2], output[3]), (0, 0, 255), 2)
                    else:
                        cv2.rectangle(image, (output[0], output[1]), (output[2], output[3]), (0, 255, 0), 2)
                    cv2.putText(image, str(output[-1]), (output[0], output[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 1)
            """图像及数据显示"""
            fps.update()
            cv2.putText(image, f"FPS:{int(fps.get())}", (850, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
            cv2.putText(image, f"BAT:{battery}%", (850, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
            cv2.putText(image, f"WIFI:{wifi_strength}%", (850, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
            cv2.imshow('UAV', image)
            """飞行控制"""
            if time.perf_counter() - x_err_time > 0.3:
                command[1] = 0
            fly.control(command)
            """绘制航迹"""
            img = np.zeros((1000, 1000, 3), np.uint8)
            drawPoints(img, points)  # 画轨迹点
            drawArrows(img, points[-1], yaw)  # 画箭头
            cv2.imshow("track", img)
            cv2.waitKey(1)
            if raw_frame.time_base < 1.0 / 40:
                time_base = 1.0 / 40
            else:
                time_base = raw_frame.time_base
            frame_skip = int((time.perf_counter() - start_time) / time_base)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置环境变量
    """创建进程共享列表"""
    frame = Manager().list()  # 当前图像像素列表
    obj_res = Manager().list()  # 物体识别结果
    face_res = Manager().list()  # 人脸识别结果
    target_res = Manager().list()  # 目标检测结果
    pose_res = Manager().list()  # 姿态检测结果
    depth_res = Manager().list()  # 深度估计结果
    """进程创建"""
    Data_processing = Process(target=data_processing, args=(frame, obj_res, face_res, target_res, pose_res, depth_res))  # 数据综合处理进程
    Obj_detector = Process(target=object_detector, args=(frame, obj_res))  # 物体检测进程
    Face_detector = Process(target=face_detector, args=(frame, face_res))  # 人脸检测进程
    Human_track = Process(target=human_track, args=(frame, target_res))  # 人类跟踪进程
    Pose_estimation = Process(target=pose_estimation, args=(frame, pose_res))  # 姿态检测进程
    Depth_estimation = Process(target=depth_estimation, args=(frame, depth_res))  # 深度估计进程
    """启动子进程"""
    Data_processing.start()  # 数据综合处理进程
    Obj_detector.start()  # 物体检测进程
    # Face_detector.start()  # 人脸检测进程
    Human_track.start()  # 人类跟踪进程
    # Pose_estimation.start()  # 姿态检测进程
    # Depth_estimation.start()  # 深度估计进程
    """等待进程结束"""
    Data_processing.join()
