import time
import keyboard
import tellopy


class Uav(object):
    def __init__(self):
        self.drone = tellopy.Tello()
        self.drone.connect()
        self.drone.wait_for_connection(60.0)
        keyboard.hook(self.pressed_keys)  # 读取键盘按键
        self.key_data = [0] * 4
        self.SPEED = 1

    def rc_control(self, *rc_data):
        self.drone.set_throttle(rc_data[0])
        self.drone.set_yaw(rc_data[1])
        self.drone.set_pitch(rc_data[2])
        self.drone.set_roll(rc_data[3])

    def start_moter(self):  # 启动螺旋桨
        self.rc_control(-1, 1, -1, -1)
        time.sleep(0.3)
        self.rc_control(0, 0, 0, 0)

    def pressed_keys(self, e):
        if e.event_type == 'down':  # 按下置1，松开置0
            data = 1
        else:
            data = 0

        if e.name == "w":  # w
            self.key_data[0] = data
        elif e.name == "s":  # s
            self.key_data[0] = -data
        elif e.name == "a":  # a
            self.key_data[1] = -data
        elif e.name == "d":  # d
            self.key_data[1] = data
        elif e.name == "up":  # up
            self.key_data[2] = data
        elif e.name == "down":  # down
            self.key_data[2] = -data
        elif e.name == "left":  # left
            self.key_data[3] = -data
        elif e.name == "right":  # right
            self.key_data[3] = data

        elif e.name == "tab":  # tab 起飞
            self.drone.takeoff()
        elif e.name == "space":  # space 降落
            self.drone.land()
        elif e.name == "enter":  # enter 前空翻
            self.drone.flip_forward()

    @staticmethod
    def limit(value, value_limit):
        if value > value_limit:
            value = value_limit
        elif value < -value_limit:
            value = -value_limit
        return value

    def control(self, auto_data):
        real_data = [0] * 4
        for i in range(4):
            real_data[i] = auto_data[i] + self.key_data[i] * self.SPEED
        self.rc_control(real_data[0], real_data[1], real_data[2], real_data[3])  # 发送控制数据


if __name__ == '__main__':
    import wifi_init
    WIFI = wifi_init.wifi()  # 实例化wifi类
    wifi_init.connect(WIFI)

    def handler(event, sender, data, **args):
        drone = sender
        if event is drone.EVENT_FLIGHT_DATA:
            print(data)

    fly = Uav()
    fly.drone.subscribe(fly.drone.EVENT_FLIGHT_DATA, handler)
    while True:
        fly.control([0] * 4)
        time.sleep(0.1)
