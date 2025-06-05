import numpy as np
from sklearn.cluster import DBSCAN

def points_in_frustum(lidar_xyz: np.array,
                      projected_uv: np.array,
                      bbox: tuple,
                      image_shape: tuple) -> np.ndarray:
    """
    Select LiDAR points whose projected image coords (u, v) lie inside the bbox.

    Arguments:
        lidar_xyz (np.ndarray[N,3]): LiDAR points in sensor frame.
        projected_uv (np.ndarray[N,2]): Corresponding pixel coords (u,v) for each LiDAR point.
        bbox (xmin, ymin, xmax, ymax): YOLO bounding box in pixels.
        image_shape (H, W): Height and width of the image (for clamping/filtering).
    Returns:
        idxs (np.ndarray[K], dtype=int): Indices of points within the frustum.
    """ 

    xmin, ymin, xmax, ymax = bbox
    H, W = image_shape

    u = projected_uv[:, 0]
    v = projected_uv[:, 1]

    # Points must lie inside the image
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # ...and inside the bounding box
    mask &= (u >= xmin) & (u <= xmax) & (v >= ymin) & (v <= ymax)

    return np.where(mask)[0]

def cluster_frustum_points(frustum_xyz: np.ndarray,
                           eps: float = 0.3,
                           min_samples: int = 10):
    """Cluster 3D points within a frustum using DBSCAN and return the largest cluster.

    Args:
        frustum_xyz (np.ndarray[K,3]): Points in the frustum in 3D space.
        eps (float): DBSCAN ``eps`` parameter in metres.
        min_samples (int): Minimum number of points required to form a cluster.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ``cluster_mask`` selecting the
            largest cluster, the centroid of that cluster and the clustered
            points themselves.
    """

    if frustum_xyz.shape[0] == 0:
        return np.zeros(0, dtype=bool), np.array([0.0, 0.0, 0.0]), frustum_xyz

    if frustum_xyz.shape[0] < min_samples:
        centroid = np.mean(frustum_xyz, axis=0)
        mask = np.ones(frustum_xyz.shape[0], dtype=bool)
        return mask, centroid, frustum_xyz

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(frustum_xyz)
    labels = clustering.labels_

    unique_labels = set(labels)
    unique_labels.discard(-1)

    if not unique_labels:
        centroid = np.mean(frustum_xyz, axis=0)
        mask = np.ones(frustum_xyz.shape[0], dtype=bool)
        return mask, centroid, frustum_xyz

    best_label = max(unique_labels, key=lambda lbl: np.sum(labels == lbl))
    cluster_mask = labels == best_label
    cluster_points = frustum_xyz[cluster_mask]
    centroid = np.mean(cluster_points, axis=0)

    return cluster_mask, centroid, cluster_points