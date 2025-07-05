# combined_tracker.py

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = areaA + areaB - interArea
    return interArea/union if union>0 else 0.0

def appearance_feature(frame, box):
    x1,y1,x2,y2 = map(int, box)
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros(3)
    # mean‐RGB normalized
    return patch.reshape(-1,3).mean(axis=0) / 255.0

class Track:
    __slots__ = ("id", "bbox", "feat", "cluster", "missed")

    def __init__(self, tid, bbox, feat, cluster):
        self.id = tid
        self.bbox = bbox  # [x1,y1,x2,y2]
        self.feat = feat  # appearance descriptor
        self.cluster = cluster  # 3-D centroid of frustum points
        self.missed = 0  # frames since last match

class CombinedTracker:
    def __init__(self, iou_weight=0.4, app_weight=0.4, cluster_weight=0.2,
                 dist_thresh=0.7, max_missed=5):
        """Multi-cue tracker using IoU, appearance and cluster distance.

        The three weight parameters determine the influence of each cost
        component when matching detections to existing tracks.
        """

        self.iou_w = iou_weight
        self.app_w = app_weight
        self.clust_w = cluster_weight
        self.thresh = dist_thresh
        self.max_missed = max_missed
        self.next_id = 0
        self.tracks = []

    def _compute_cost_matrix(self, boxes, feats, clusters):
        N = len(self.tracks)
        M = len(boxes)
        cost = np.zeros((N, M), dtype=float)
        for i, tr in enumerate(self.tracks):
            for j, (b, f, c) in enumerate(zip(boxes, feats, clusters)):
                c_iou = 1.0 - iou(tr.bbox, b)
                c_app = np.linalg.norm(tr.feat - f)
                c_clu = np.linalg.norm(tr.cluster - c)
                cost[i, j] = (
                    self.iou_w * c_iou +
                    self.app_w * c_app +
                    self.clust_w * c_clu
                )
        return cost

    def update(self, det_results, frame, clusters):
        """Update tracker state with new detections.

        Args:
            det_results: Output of ``YOLO(frame)``.
            frame: BGR image used for the appearance descriptor.
            clusters: List of 3-D cluster centroids corresponding to each
                detection bounding box.

        Returns:
            A list of dictionaries with ``id``, ``box`` and ``center`` for
            visualization.
        """
        # --- 1) extract detections + features
        boxes = []
        feats = []
        for det in det_results.boxes:
            x1, y1, x2, y2 = det.xyxy[0].tolist()
            boxes.append([x1, y1, x2, y2])
            feats.append(appearance_feature(frame, (x1, y1, x2, y2)))

        if not self.tracks:
            for b, f, c in zip(boxes, feats, clusters):
                self.tracks.append(Track(self.next_id, b, f, c))
                self.next_id += 1
        else:
            cost = self._compute_cost_matrix(boxes, feats, clusters)
            # Hungarian assignment
            row_idx, col_idx = linear_sum_assignment(cost)

            assigned_tracks = set()
            assigned_dets   = set()
            # --- 2) match within threshold
            for r, c in zip(row_idx, col_idx):
                if cost[r,c] < self.thresh:
                    tr = self.tracks[r]
                    tr.bbox = boxes[c]
                    tr.feat = feats[c]
                    tr.cluster = clusters[c]
                    tr.missed = 0
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

            # --- 3) mark unmatched tracks
            for i, tr in enumerate(self.tracks):
                if i not in assigned_tracks:
                    tr.missed += 1
            # delete stale
            self.tracks = [tr for tr in self.tracks if tr.missed <= self.max_missed]

            # --- 4) create new tracks for unmatched detections
            for j, (b, f, c) in enumerate(zip(boxes, feats, clusters)):
                if j not in assigned_dets:
                    self.tracks.append(Track(self.next_id, b, f, c))
                    self.next_id += 1

        # --- 5) return simple list for visualization
        out = []
        for tr in self.tracks:
            x1,y1,x2,y2 = map(int, tr.bbox)
            cx, cy      = int((x1+x2)/2), int((y1+y2)/2)
            out.append({"id":tr.id, "box":[x1,y1,x2,y2], "center":(cx,cy)})
        return out
