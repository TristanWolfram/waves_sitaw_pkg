import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import os
from scipy.optimize import linear_sum_assignment

video_name = "blueboat_stream3"
PATH_TO_VIDEO = f"rbi_perception_pkg/testing/videos/{video_name}.mp4"
PATH_OUTPUT = f"rbi_perception_pkg/testing/videos/{video_name}_track_custom.mp4"

# combined_tracker.py

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
    __slots__ = ("id","bbox","feat","missed")
    def __init__(self, tid, bbox, feat):
        self.id     = tid
        self.bbox   = bbox    # [x1,y1,x2,y2]
        self.feat   = feat    # appearance descriptor
        self.missed = 0       # frames since last match

class CombinedTracker:
    def __init__(self, iou_weight=0.5, dist_thresh=0.7, max_missed=5):
        """
        iou_weight: weight for IoU vs. appearance distance
        dist_thresh: maximum allowed combined cost to accept a match
        max_missed: how many frames to keep “lost” tracks alive
        """
        self.iou_w      = iou_weight
        self.thresh     = dist_thresh
        self.max_missed = max_missed
        self.next_id    = 0
        self.tracks     = []

    def _compute_cost_matrix(self, boxes, feats):
        N = len(self.tracks)
        M = len(boxes)
        cost = np.zeros((N, M), dtype=float)
        for i, tr in enumerate(self.tracks):
            for j, (b, f) in enumerate(zip(boxes, feats)):
                c_iou = 1.0 - iou(tr.bbox, b)
                c_app = np.linalg.norm(tr.feat - f)
                cost[i, j] = self.iou_w * c_iou + (1-self.iou_w) * c_app
        return cost

    def update(self, det_results, frame):
        """
        det_results: the YOLO results object (from model(frame)[0])
        frame: BGR image (for appearance_feature)
        Returns list of dicts {"id", "box", "center"}
        """
        # --- 1) extract detections + features
        boxes = []
        feats = []
        for det in det_results.boxes:
            x1,y1,x2,y2 = det.xyxy[0].tolist()
            boxes.append([x1,y1,x2,y2])
            feats.append(appearance_feature(frame, (x1,y1,x2,y2)))

        if not self.tracks:
            # initialize all as new tracks
            for b,f in zip(boxes, feats):
                self.tracks.append(Track(self.next_id, b, f))
                self.next_id += 1
        else:
            cost = self._compute_cost_matrix(boxes, feats)
            # Hungarian assignment
            row_idx, col_idx = linear_sum_assignment(cost)

            assigned_tracks = set()
            assigned_dets   = set()
            # --- 2) match within threshold
            for r, c in zip(row_idx, col_idx):
                if cost[r,c] < self.thresh:
                    tr = self.tracks[r]
                    tr.bbox   = boxes[c]
                    tr.feat   = feats[c]
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
            for j, (b,f) in enumerate(zip(boxes, feats)):
                if j not in assigned_dets:
                    self.tracks.append(Track(self.next_id, b, f))
                    self.next_id += 1

        # --- 5) return simple list for visualization
        out = []
        for tr in self.tracks:
            x1,y1,x2,y2 = map(int, tr.bbox)
            cx, cy      = int((x1+x2)/2), int((y1+y2)/2)
            out.append({"id":tr.id, "box":[x1,y1,x2,y2], "center":(cx,cy)})
        return out


def main():

    # 1) Load YOLO model & CombinedTracker
    model   = YOLO("yolo12n.pt")
    tracker = CombinedTracker(
        iou_weight=0.7, 
        dist_thresh=0.7, 
        max_missed=5
    )
    track_histories = {}
    max_history     = 30

    # 2) Open video
    cap = cv2.VideoCapture(PATH_TO_VIDEO)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {PATH_TO_VIDEO}")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 3) Prepare writer if requested
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(PATH_OUTPUT, fourcc, fps, (width, height))

    # 4) Frame loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4.1) Run YOLO detection
        results = model(frame)[0]

        # 4.2) Update tracker
        tracks = tracker.update(results, frame)

        # 4.3) Purge old histories
        active_ids = {t["id"] for t in tracks}
        for tid in list(track_histories):
            if tid not in active_ids:
                del track_histories[tid]

        # 4.4) Draw boxes + build histories
        H, W = frame.shape[:2]
        for t in tracks:
            tid = t["id"]
            x1,y1,x2,y2 = t["box"]
            cx, cy     = t["center"]

            # update history
            hist = track_histories.setdefault(tid, [])
            hist.append((cx, cy))
            if len(hist) > max_history:
                hist.pop(0)

            # draw box and ID
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,128,255), 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # 4.5) Draw trajectories
        for tid, hist in track_histories.items():
            if len(hist) < 2:
                continue
            pts = np.array(hist, dtype=np.int32).reshape(-1,1,2)
            color = ((tid * 37) % 255, (tid * 91) % 255, (tid * 53) % 255)
            cv2.polylines(frame, [pts], False, color, 2)

        # 4.6) Output
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if writer:
            writer.write(frame)

    # 5) Cleanup
    cap.release()
    if writer:
        writer.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

