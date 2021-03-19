import time


class FPS:  # ����֡��
    def __init__(self):
        self.nbf = 0
        self.fps = 0
        self.start = 0

    def update(self):  # ÿ10֡����һ��֡��
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
