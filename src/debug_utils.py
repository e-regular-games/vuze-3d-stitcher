
from matplotlib import pyplot as plt
import math
import numpy as np
import time
import threading

class Debug:
    def __init__(self, options=None):
        self.display = options.display if options is not None else {}
        self.verbose = options.verbose if options is not None else False
        self.enable_threads = len(self.display) == 0
        self.enable_threads_high_mem = len(self.display) == 0
        self._window_title = ''
        self._figures = {}
        self._subplot = (1, 1, 1)
        self._msgs = []
        self._msg_pause = False
        self._perfs = {}

    def log_pause(self):
        self._msg_pause = True

    def log_resume(self):
        self._msg_pause = False
        if self.verbose and len(self._msgs) > 0:
            for m in self._msgs:
                print(*m)
            self._msgs = []

    def log(self, *args):
        if not self.verbose:
            return

        if self._msg_pause:
            self._msgs.append([str(a) for a in args])
        else:
            print(*args)

    def enable(self, opt):
        return opt in self.display and self.display[opt]

    def figure(self, id, reset=False):
        if id not in self._figures or reset:
            self._figures[id] = plt.figure()
            self._figures[id].canvas.manager.set_window_title(self._window_title + ' - ' + id)
        return self._figures[id]

    def subplot(self, id, projection=None):
        return self.figure(id) \
                   .add_subplot(self._subplot[0], self._subplot[1], \
                                self._subplot[2], projection=projection)

    # create a new figure window for each figure
    def window(self, prefix):
        # avoid the full clone because we want to reset the figures and subplot
        d = self.clone()
        d._figures = {}
        d._subplot = (1, 1, 1)
        d._window_title = self._window_title + ' - ' + prefix
        return d

    def set_subplot(self, h, w, idx):
        d = self.clone()

        # parent row and column
        rp = int((self._subplot[2] - 1) / self._subplot[1])
        cp = (self._subplot[2] - 1) % self._subplot[1]

        # child row and column
        rc = int((idx - 1) / w)
        cc = (idx - 1) % w

        rowcells = h * w * self._subplot[1]
        i = rp * rowcells + rc * w * self._subplot[1] + cp * w + cc + 1

        d._subplot = (self._subplot[0] * h, self._subplot[1] * w, i)
        return d

    def clone(self):
        d = Debug(self)
        d._figures = self._figures
        d._subplot = self._subplot
        d._window_title = self._window_title
        d.enable_threads = self.enable_threads
        d.enable_threads_high_mem = self.enable_threads_high_mem
        d._msgs = self._msgs
        d._msg_pause = self._msg_pause
        return d

    def perf(self, name):
        t = str(threading.get_ident())
        if not t in self._perfs:
            self._perfs[t] = {}

        if name in self._perfs[t]:
            self.log('performance', name, t, time.perf_counter() - self._perfs[t][name])
            del self._perfs[t][name]
        else:
            self._perfs[t][name] = time.perf_counter()

    def none(self):
        return Debug()

def show_polar_plot(polar_a, polar_b, label=None):
    plt.figure(label) if label is not None else plt.figure()

    ra  = polar_a[..., 2] if polar_a.shape[-1] > 2 else np.array([1], np.float32)
    xa = ra * np.sin(polar_a[..., 0]) * np.cos(polar_a[..., 1])
    ya = ra * np.sin(polar_a[..., 0]) * np.sin(polar_a[..., 1])
    za = ra * np.cos(polar_a[..., 0])

    ax = plt.axes(projection ='3d')
    ax.plot3D(xa, ya, za, 'bo', markersize=1)

    rb  = polar_b[..., 2] if polar_b.shape[-1] > 2 else np.array([1], np.float32)
    xb = rb * np.sin(polar_b[..., 0]) * np.cos(polar_b[..., 1])
    yb = rb * np.sin(polar_b[..., 0]) * np.sin(polar_b[..., 1])
    zb = rb * np.cos(polar_b[..., 0])

    r = max(np.max(ra), np.max(rb))
    ax.plot3D(xb, yb, zb, 'ro', markersize=0.5)
    ax.axes.set_xlim3d(left=-r, right=r)
    ax.axes.set_ylim3d(bottom=-r, top=r)
    ax.axes.set_zlim3d(bottom=-r, top=r)

def show_polar_points(polar, label=None):
    plt.figure(label) if label is not None else plt.figure()
    ra  = polar[..., 2] if polar.shape[-1] > 2 else 1
    xa = ra * np.sin(polar[..., 0]) * np.cos(polar[..., 1])
    ya = ra * np.sin(polar[..., 0]) * np.sin(polar[..., 1])
    za = ra * np.cos(polar[..., 0])

    r = np.max(ra)
    ax = plt.axes(projection ='3d')
    ax.plot3D(xa, ya, za, 'b.', markersize=0.5)
    ax.axes.set_xlim3d(left=-r, right=r)
    ax.axes.set_ylim3d(bottom=-r, top=r)
    ax.axes.set_zlim3d(bottom=-r, top=r)
