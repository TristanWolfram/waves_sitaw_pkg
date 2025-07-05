import numpy as np
"""Helper methods for point cloud / image processing."""

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

def cluster_frustum_points(
    frustum_xyz: np.ndarray, bins: int = 60, sigma: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Estimate a cluster centroid and axis line using a depth histogram.

    The points inside the provided frustum are histogrammed along the Z-axis.
    The highest histogram bin is selected and all points within ``sigma`` bins
    around this peak are averaged to obtain the cluster centroid.

    Args:
        frustum_xyz: Points belonging to a single detection frustum ``(N, 3)``.
        bins: Number of histogram bins along the depth axis.
        sigma: Half-width (in bins) of the neighborhood around the peak.

    Returns:
        ``cluster_mask`` selecting the points used for the centroid,
        the centroid itself, the selected points and the axis line
        ``(left_point, right_point)``.
    """

    if frustum_xyz.size == 0:
        return (
            np.zeros(0, dtype=bool),
            np.array([0.0, 0.0, 0.0]),
            frustum_xyz,
            (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        )

    depths = frustum_xyz[:, 2]
    hist, bin_edges = np.histogram(depths, bins=bins)

    peak_idx = int(np.argmax(hist))
    start = max(peak_idx - sigma, 0)
    end = min(peak_idx + sigma + 1, len(hist))

    z_min = bin_edges[start]
    z_max = bin_edges[end]

    mask = (depths >= z_min) & (depths < z_max)
    cluster_points = frustum_xyz[mask]

    if cluster_points.size == 0:
        centroid = frustum_xyz.mean(axis=0)
        mask = np.ones(frustum_xyz.shape[0], dtype=bool)
        cluster_points = frustum_xyz
    else:
        centroid = cluster_points.mean(axis=0)

    # ------------------------------------------------------------------
    # Estimate axis line by expanding left/right until histogram gap
    # (bin with zero count) is found. The Z values at those gaps are
    # converted back to actual points by taking the closest points
    # along the depth dimension.
    left_idx = peak_idx
    while left_idx > 0 and hist[left_idx] > 0:
        left_idx -= 1
    if hist[left_idx] > 0:
        z_left = bin_edges[0]
    else:
        z_left = bin_edges[left_idx + 1]

    right_idx = peak_idx
    while right_idx < len(hist) - 1 and hist[right_idx] > 0:
        right_idx += 1
    if hist[right_idx] > 0:
        z_right = bin_edges[-1]
    else:
        z_right = bin_edges[right_idx]

    left_point = frustum_xyz[np.argmin(np.abs(depths - z_left))]
    right_point = frustum_xyz[np.argmin(np.abs(depths - z_right))]

    return mask, centroid, cluster_points, (left_point, right_point)