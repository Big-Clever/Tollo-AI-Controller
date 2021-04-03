import cv2
import uav
import time
import wifi_init
import paddlehub as hub
import os
from multiprocessing import Process, Queue, Manager
import numpy as np
import math
from math import pi, atan2, degrees, sqrt
import pid
import deepsort
import copy
import av
import FPS
import gc
from CameraMorse import CameraMorse
import DrawPose
import operator


def distance(A, B):
    """计算点A到点B之间的距离"""
    return int(sqrt((B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2))


def angle(A, B, C):
    """计算线段(A,B)与线段(B,C)夹角"""
    return degrees(atan2(C[1] - B[1], C[0] - B[0]) - atan2(A[1] - B[1], A[0] - B[0])) % 360


def vertical_angle(A, B):
    """计算线段(A,B)与纵轴的夹角"""
    return degrees(atan2(B[1] - A[1], B[0] - A[0]) - pi / 2)


def quaternion2yaw(q0, q1, q2, q3):
    """四元数转偏航角"""
    degree = math.pi / 180
    siny = 2.0 * (q0 * q3 + q1 * q2)
    cosy = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    return int(math.atan2(siny, cosy) / degree)


def drawPoints(img, points):
    """绘制航迹"""
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 0, 255), cv2.FILLED)
    cv2.putText(img, f"({((points[-1][0] - 500) / 10):.1f}, {-((points[-1][1] - 500) / 10):.1f})m",
                (int(points[-1][0]) + 10, int(points[-1][1]) + 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)


def drawArrows(img, point, yaw):
    """绘制箭头"""
    length = 12
    angle = 40
    yaw += 90
    fpoint = [int(point[0] + length * math.cos(math.radians(yaw - 180))),
              int(point[1] + length * math.sin(math.radians(yaw - 180)))]
    bpoint = [int(point[0] + length * math.cos(math.radians(yaw)) / 6),
              int(point[1] + length * math.sin(math.radians(yaw)) / 6)]
    lpoint = [int(point[0] + length * math.cos(math.radians(yaw - angle))),
              int(point[1] + length * math.sin(math.radians(yaw - angle)))]
    rpoint = [int(point[0] + length * math.cos(math.radians(yaw + angle))),
              int(point[1] + length * math.sin(math.radians(yaw + angle)))]
    area = np.array([fpoint, lpoint, bpoint, rpoint])
    cv2.fillConvexPoly(img, area, (0, 255, 0))


def object_detector(share):
    """物体检测进程"""
    # 加载模型，可选模型 ssd_mobilenet_v1_pascal 和 yolov3_mobilenet_v1_coco2017
    detector = hub.Module(name="yolov3_mobilenet_v1_coco2017")
    while share.get("frame") is None:
        time.sleep(0.1)
    while True:
        res = detector.object_detection(
            images=[share["frame"]],
            batch_size=1,
            use_gpu=True,
            score_thresh=0.4,
            visualization=False
        )
        share["obj"] = res


def face_detector(share):
    """人脸检测进程"""
    face_detector = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_640")
    while share.get("frame") is None:
        time.sleep(0.1)
    while True:
        res = face_detector.face_detection(
            images=[share["frame"]],
            use_gpu=False,
            visualization=False,
            confs_threshold=0.8)
        share["face"] = res


def human_track(share):
    """目标跟踪进程"""
    human_track = deepsort.DeepSort(use_gpu=True, threshold=0.2)
    while share.get("frame") is None:
        time.sleep(0.1)
    while True:
        res = human_track.update(share["frame"])
        share["target"] = res


def pose_estimation(share):
    """骨骼关键点检测进程"""
    frame_last = [0] * 3
    # 可选模型 openpose_body_estimation 和 openpose_hands_estimation
    pose_estimation = hub.Module(name="openpose_body_estimation")
    while share.get("person_frame") is None:
        time.sleep(0.1)
    while True:
        if frame_last[2] != share["person_frame"][2]:
            frame_last[2] = share["person_frame"][2]
            person_data = share["person_frame"]
            res = pose_estimation.predict(
                img=person_data[0],
                # scale=[0.5, 1.0, 1.5, 2.0],  # 识别手部关键点时,使用图片的不同尺度
                visualization=False)
            if len(res["candidate"]) > 0 and len(res["subset"]) > 0:
                share["pose"] = [res, person_data[1], person_data[2]]
        else:
            time.sleep(0.01)


def depth_estimation(share):
    """深度估计进程"""
    model = hub.Module(name='MiDaS_Small', use_gpu=True)
    while share.get("frame") is None:
        time.sleep(0.1)
    while True:
            outputs = model.depth_estimation(
                images=[share["frame"]],
                paths=None,
                batch_size=1)
            res = outputs[0]
            share["depth"] = res


pos_x, pos_y = 500, 500  # 轨迹起始点坐标
points = [[500, 500]]
yaw = 0  # 偏航角
battery = 0  # 电量
wifi_strength = 0  # 信号强度
fly_mode = 1
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
    global pos_x, pos_y, yaw, battery, wifi_strength, points, fly_mode
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
        fly_mode = data.fly_mode


def data_processing(share):
    """数据综合处理进程"""
    WIFI = wifi_init.wifi()  # 实例化wifi类
    wifi_init.connect(WIFI)

    global pos_x, pos_y, yaw, battery, wifi_strength, points, mouse_x, mouse_y, target_index, fly_mode
    fly = uav.Uav()
    drone = fly.drone
    drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
    drone.subscribe(drone.EVENT_LOG_DATA, handler)
    cv2.namedWindow("UAV")
    cv2.setMouseCallback("UAV", capture_mouse_event)
    draw_pose = DrawPose.DrawPose()
    cm = CameraMorse(display=False)

    cm.define_command("--", fly.start_moter)
    cm.define_command("..", drone.takeoff)
    cm.define_command(".-", drone.land)

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

    face_res_last = []
    target_res_last = []
    pose_res_last = [0] * 3

    neck_length_2m = 72  # 此处为2m时脖子长度
    target_distance = 2
    exp_distance = 2

    width, height = 960, 720  # 图像宽度、高度
    command = [0.0] * 4
    Yaw_pid = pid.PID(0.002, 0, 0.0005, 0, 0.5)
    Pitch_pid = pid.PID(0.5, 0, 0.04, 0, 0.6)
    Height_pid = pid.PID(0.002, 0, 0, 0, 0.5)

    pose_time = 0
    yaw_err_time = 0
    pitch_err_time = 0
    height_err_time = 0
    roll_err_time = 0

    body_kp_id_to_name = {
        0: "Nose",
        1: "Neck",
        2: "RShoulder",
        3: "RElbow",
        4: "RWrist",
        5: "LShoulder",
        6: "LElbow",
        7: "LWrist",
        8: "RHip",
        9: "RKnee",
        10: "RAnkle",
        11: "LHip",
        12: "LKnee",
        13: "LAnkle",
        14: "REye",
        15: "LEye",
        16: "REar",
        17: "LEar"}

    body_kp_name_to_id = {v: k for k, v in body_kp_id_to_name.items()}

    def check_var_exist(*args):
        for arg in args:
            if arg is None:
                return False
        return True

    def get_body_kp(kp_name="Neck"):
        """返回名为“kp_name”的关键点坐标(从0开始)，如果没有检测到关键点，则返回None"""
        try:
            index = posedata["subset"][0][body_kp_name_to_id[kp_name]]
            point = posedata["candidate"][int(index)][0:2]
            return point
        except:
            return None

    while True:
        for raw_frame in container.decode(video=0):
            if 0 < frame_skip:
                frame_skip = frame_skip - 1
                continue
            start_time = time.perf_counter()
            """图像获取"""
            raw_image = cv2.cvtColor(np.array(raw_frame.to_image()), cv2.COLOR_RGB2BGR)
            share["frame"] = raw_image
            image = copy.deepcopy(raw_image)
            """物体检测数据处理"""
            obj_res = share.get("obj")
            if obj_res is not None:
                for data in obj_res[0]["data"]:
                    if data['label'] != "person":
                        left = int(data["left"])
                        top = int(data["top"])
                        right = int(data["right"])
                        bottom = int(data["bottom"])
                        cv2.rectangle(image, (left, top), (right, bottom), (255, 100, 100), 2)
                        cv2.putText(image, f"{data['label']}", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (100, 255, 100), 1)
            """人脸识别数据处理"""
            face_res = share.get("face")
            if face_res is not None:
                faceListC = []
                faceListSize = []
                for face in face_res[0]["data"]:
                    left = int(face["left"])
                    top = int(face["top"])
                    right = int(face["right"])
                    bottom = int(face["bottom"])
                    faceListC.append([(right + left) // 2, (bottom + top) // 2])
                    faceListSize.append(right + bottom - left - top)
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)  # 在图片中标注人脸，并显示
                """人脸跟踪，输出偏航控制数据"""
                # if len(faceListSize) != 0 and not np.all(operator.eq(face_res_last, face_res)):
                #     face_res_last = face_res
                #     i = faceListSize.index(max(faceListSize))
                #     x_err = faceListC[i][0] - width / 2
                #     command[1] = Yaw_pid.update(x_err)
                #     yaw_err_time = time.perf_counter()
            """目标跟踪数据处理"""
            target_res = share.get("target")
            if target_res is not None:
                for output in target_res:
                    if mouse_x != 0:
                        if output[0] < mouse_x < output[2] and output[1] < mouse_y < output[3]:
                            target_index = output[-1]
                            mouse_x = 0
                    """发现跟踪目标"""
                    if target_index == output[-1]:
                        cv2.rectangle(image, (output[0], output[1]), (output[2], output[3]), (0, 0, 255), 2)  # 红色框出目标
                        cv2.putText(image, f"{target_distance:.1f}m", (output[2] - 50, output[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                        """传递目标图像,用以检测骨骼关键点"""
                        target_size = np.where(output >= 0, output, 0)
                        if target_size[2] >= raw_image.shape[1]:
                            target_size[2] = raw_image.shape[1] - 1
                        if target_size[3] >= raw_image.shape[0]:
                            target_size[3] = raw_image.shape[0] - 1
                        person_pic = raw_image[target_size[1]:target_size[3], target_size[0]:target_size[2]]
                        scale = math.sqrt(30000 / ((target_size[3] - target_size[1]) * (target_size[2] - target_size[0])))  # 将图片缩小至30000个像素
                        if scale < 1:
                            person_pic = cv2.resize(person_pic, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                        share["person_frame"] = [person_pic, target_size, scale]
                        """目标跟踪控制"""
                        if not np.all(operator.eq(target_res_last, target_res[0])):
                            target_res_last = target_res[0]
                            "输出偏航控制数据"
                            x_err = (output[0] + output[2]) / 2 - width / 2
                            command[1] = Yaw_pid.update(x_err)
                            yaw_err_time = time.perf_counter()
                    else:  # 绿色框出所有人
                        cv2.rectangle(image, (output[0], output[1]), (output[2], output[3]), (0, 255, 0), 2)
            """骨骼关键点数据处理"""
            pose_res = share.get("pose")
            if pose_res is not None:
                posedata = pose_res[0]
                target_size = pose_res[1]
                scale = pose_res[2]
                if scale < 1:
                    posedata["candidate"][:, 0:2] = posedata["candidate"][:, 0:2] / scale
                posedata["candidate"][:, 0] += target_size[0]
                posedata["candidate"][:, 1] += target_size[1]
                if time.perf_counter() - pose_time < 0.2:
                    image = draw_pose(image, posedata["candidate"], posedata["subset"])
                """若为新数据，则进行处理"""
                if pose_res_last[2] != pose_res[2]:
                    pose_res_last[2] = pose_res[2]
                    pose_time = time.perf_counter()
                    clavicle_index = posedata["subset"][0][1]  # 锁骨中心点索引
                    ear_index = [x for x in posedata["subset"][0][16:18] if x != -1]  # 检测到的耳朵索引
                    """计算目标距离"""
                    if clavicle_index != -1 and len(ear_index) > 0:  # 若锁骨索引和耳朵索引存在
                        clavicle_y = posedata["candidate"][int(clavicle_index)][1]  # 锁骨高度坐标
                        ear_y = np.mean([posedata["candidate"][int(x)][1] for x in ear_index])  # 耳朵高度坐标
                        neck_length = clavicle_y - ear_y
                        if neck_length > 0:
                            target_distance = neck_length_2m / neck_length + target_distance * 0.5  # 计算距离
                            """动作识别"""
                            RShoulder = get_body_kp("RShoulder")
                            RElbow = get_body_kp("RElbow")
                            RWrist = get_body_kp("RWrist")
                            LShoulder = get_body_kp("LShoulder")
                            LElbow = get_body_kp("LElbow")
                            LWrist = get_body_kp("LWrist")
                            if check_var_exist(RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist):  # 若左臂右臂都检测到
                                if distance(RShoulder, RWrist) < neck_length and distance(LShoulder, LWrist) < neck_length:  # 若左右手肘在左右肩附近
                                    if RElbow[1] - RShoulder[1] < neck_length * 1.2 and LElbow[1] - LShoulder[1] < neck_length * 1.2:
                                        exp_distance += 0.05
                                    else:
                                        exp_distance -= 0.05
                            if check_var_exist(RShoulder, RElbow, RWrist):  # 若检测到右臂
                                if abs(RShoulder[1] - RWrist[1]) < neck_length / 2 and abs(RShoulder[1] - RElbow[1]) < neck_length * 1.2:
                                    command[3] = (RWrist[0] - RShoulder[0]) / neck_length / 5
                                    roll_err_time = time.perf_counter()
                        "输出前后控制数据"
                        pitch_err = target_distance - exp_distance
                        command[2] = Pitch_pid.update(pitch_err)
                        pitch_err_time = time.perf_counter()
                    if clavicle_index != -1:  # 若锁骨索引存在
                        clavicle_y = posedata["candidate"][int(clavicle_index)][1] + target_size[1]  # 锁骨高度坐标
                        "输出高度控制数据"
                        height_err = height / 2 - clavicle_y
                        command[0] = Height_pid.update(height_err)
                        height_err_time = time.perf_counter()
            """图像及数据显示"""
            fps.update()
            cv2.putText(image, f"FPS:{int(fps.get())}", (850, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
            cv2.putText(image, f"BAT:{battery}%", (850, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
            cv2.putText(image, f"WIFI:{wifi_strength}%", (850, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
            cv2.putText(image, f"MODE:{fly_mode}", (850, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
            cv2.putText(image, f"EXP:{exp_distance}", (850, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

            cv2.imshow('UAV', image)
            """飞行控制"""
            if time.perf_counter() - yaw_err_time > 0.6:
                command[1] = 0
            if time.perf_counter() - pitch_err_time > 0.2:
                command[2] = 0
            if time.perf_counter() - height_err_time > 0.2:
                command[0] = 0
            if time.perf_counter() - roll_err_time > 0.2:
                command[3] = 0
            fly.control(command)
            """绘制航迹"""
            img = np.zeros((1000, 1000, 3), np.uint8)
            drawPoints(img, points)  # 画轨迹点
            drawArrows(img, points[-1], yaw)  # 画箭头
            cv2.imshow("track", img)
            cv2.waitKey(1)
            if fly_mode == 1:
                cm.eval(image)  # 摄像头作按键检测
            if raw_frame.time_base < 1.0 / 40:
                time_base = 1.0 / 40
            else:
                time_base = raw_frame.time_base
            frame_skip = int((time.perf_counter() - start_time) / time_base)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置环境变量
    """创建进程共享字典"""
    share = Manager().dict()  # 创建一个字典,可在多个进程间传递和共享
    """进程创建"""
    Data_processing = Process(target=data_processing, args=(share,))  # 数据综合处理进程
    Obj_detector = Process(target=object_detector, args=(share,))  # 物体检测进程
    Face_detector = Process(target=face_detector, args=(share,))  # 人脸检测进程
    Human_track = Process(target=human_track, args=(share,))  # 人类跟踪进程
    Pose_estimation = Process(target=pose_estimation, args=(share,))  # 姿态检测进程
    Depth_estimation = Process(target=depth_estimation, args=(share,))  # 深度估计进程
    """启动子进程"""
    Data_processing.start()  # 数据综合处理进程
    # Obj_detector.start()  # 物体检测进程
    # Face_detector.start()  # 人脸检测进程
    Human_track.start()  # 人类跟踪进程
    Pose_estimation.start()  # 姿态检测进程
    # Depth_estimation.start()  # 深度估计进程
    """等待进程结束"""
    while True:
        gc.collect()
        time.sleep(1)
    # Data_processing.join()
