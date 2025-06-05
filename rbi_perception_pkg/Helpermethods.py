import numpy as np

def points_in_frustum(lidar_xyz: np.array,
                      projected_uv: np.array,
                      bbox: tuple,
                      image_shape: tuple,):
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

    # Filter out points outside the image bounds
    u = projected_uv[:, 0]
    v = projected_uv[:, 1]
    valid = (u >= xmin) & (u <= xmax) & (v >= ymin) & (v <= ymax)

    # Within image -> check bounding box
    in_box = (u >= xmin) & (u <= xmax) & (v >= ymin) & (v <= ymax)

    # Combine and return
    mask = valid & in_box
    idxs = np.where(mask)[0]
    return idxs