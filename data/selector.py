import random
import math


__all__ = ['FrameSelector', 'FixedFrameSelector', 'TSNSelector']

class FrameSelector(object):

    def select(self, length, fps):
        raise NotImplementedError("function select not implemented")


class FixedFrameSelector(FrameSelector):

    def __init__(self, nframes):
        super(FixedFrameSelector, self).__init__()
        self._nframes = nframes

    def select(self, length, fps):
        step = math.floor(length / self._nframes)
        start = random.randint(0, length - (self._nframes - 1) * step - 1)
        selected = [i * step + start for i in range(self._nframes)]
        return selected


class TSNSelector(FrameSelector):

    def __init__(self, nframes, random=True):
        super(TSNSelector, self).__init__()
        self._nframes = nframes
        self._random = random

    def select(self, length, fps):
        seg = length / self._nframes
        if self._random:
            selected = [random.randint(math.floor(i*seg), math.ceil((i+1)*seg)-1) for i in range(self._nframes)]
        else:
            selected = [int((i + 0.5) * seg) for i in range(self._nframes)]
        return selected


