import numpy as np
from collections import deque

class lukaQueue:
    def __init__(self):
        self.frames = deque()
        self.averageFrame = []
        self.sumOfFrames = []


    def addFrame(self, frame):
        if len(self.frames) == 0:
            self.averageFrame = np.zeros((len(frame),len(frame[0]),len(frame[0][0])), dtype=np.uint16)
            self.sumOfFrames = np.zeros((len(frame),len(frame[0]),len(frame[0][0])), dtype=np.uint16)

        self.frames.append(frame)
        self.sumOfFrames = np.add(self.sumOfFrames, frame)
        self.newAverage()
        if len(self.frames) > 10:
            tempFrame = self.frames.popleft()
            self.sumOfFrames = np.subtract(self.sumOfFrames, tempFrame)
            self.newAverage()

    def getAverageFrame(self):
        return self.averageFrame.astype(np.uint8) #this could be optimized, but it must return a frame of type uint8

    def newAverage(self):
        self.averageFrame = np.divide(self.sumOfFrames, len(self.frames))
