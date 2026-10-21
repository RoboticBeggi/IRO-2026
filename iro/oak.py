"""OAK-D Pro W: colour + depth aligned pixel-for-pixel, and 3D measurements of objects.

Settings chosen from tests on the robot's floor (20 Oct): ROBOTICS preset, 640x400 stereo with
extended disparity (sees the near floor), IR dot projector at 50% (depth on plain sponges).
Coordinates are the camera's: x right, y down, z forward (metres).
"""

import time

import cv2
import numpy as np


class Oak:
    def __init__(self, ir=0.5, size=(640, 400)):
        import depthai as dai
        self.dai, self.size = dai, size
        self.p = dai.Pipeline()
        rgb = self.p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        left = self.p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = self.p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        st = self.p.create(dai.node.StereoDepth).build(left.requestOutput(size), right.requestOutput(size),
                                                       presetMode=dai.node.StereoDepth.PresetMode.ROBOTICS)
        st.setExtendedDisparity(True)
        st.setLeftRightCheck(True)
        st.setSubpixel(True)
        st.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        st.setOutputSize(*size)
        sync = self.p.create(dai.node.Sync)
        rgb.requestOutput(size, dai.ImgFrame.Type.BGR888i).link(sync.inputs["rgb"])
        st.depth.link(sync.inputs["depth"])
        self.q = sync.out.createOutputQueue(maxSize=2, blocking=False)
        self.p.start()
        dev = self.p.getDefaultDevice()
        dev.setIrLaserDotProjectorIntensity(ir)
        cal = dev.readCalibration()
        self.K = np.array(cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, *size))
        self.dist = np.array(cal.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))[:8]   # wide lens
        time.sleep(1.5)

    def read(self, n=3):
        """(colour BGR, depth mm). Depth is the per-pixel median of n frames (fills flicker holes)."""
        frames = []
        for _ in range(n):
            m = self.q.get()
            img = m["rgb"].getCvFrame()
            frames.append(m["depth"].getFrame().astype(np.float32))
        d = np.stack(frames)
        d[d == 0] = np.nan
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # all-empty pixels stay 0
                depth = np.nan_to_num(np.nanmedian(d, axis=0)).astype(np.uint16)
        return img, depth

    def close(self):
        self.p.stop()

    # --- geometry -------------------------------------------------------------
    def points(self, depth, mask):
        """3D points (camera frame, metres) of the valid depth pixels inside mask."""
        v, u = np.nonzero(mask & (depth > 0))
        z = depth[v, u] / 1000.0
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        return np.c_[(u - cx) * z / fx, (v - cy) * z / fy, z]

    def measure(self, depth, box, inner=0.6):
        """3D points of an object from its box: the middle `inner` part of the box (the edges mix
        with the floor), minus points far behind the object's own front surface."""
        x1, y1, x2, y2 = box
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * inner, (y2 - y1) * inner
        mask = np.zeros(depth.shape, bool)
        mask[int(cy - h / 2):int(cy + h / 2), int(cx - w / 2):int(cx + w / 2)] = True
        pts = self.points(depth, mask)
        if len(pts) < 20:
            return None
        near = np.percentile(pts[:, 2], 20)
        return pts[pts[:, 2] < near + 0.08]          # drop floor seen beside/behind the object
