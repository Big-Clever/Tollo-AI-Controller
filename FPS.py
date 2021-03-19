import time


class FPS:  # 测量帧率
    def __init__(self):
        self.nbf = 0
        self.fps = 0
        self.start = 0

    def update(self):  # 每10帧计算一次帧率
        if self.nbf % 10 == 0:
            if self.start != 0:
                self.stop = time.perf_counter()
                self.fps = 10 / (self.stop - self.start)
                self.start = self.stop
            else:
                self.start = time.perf_counter()
        self.nbf += 1

    def get(self):
        return self.fps
