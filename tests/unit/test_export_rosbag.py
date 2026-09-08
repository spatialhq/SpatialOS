import numpy as np
import open3d as o3d

from spatial_os.export.rosbag import write_rosbag
from spatial_os.schema import Frame


def _frames(n=3) -> list[Frame]:
    frames = []
    for i in range(n):
        pose = [[1, 0, 0, i * 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        frames.append(Frame(frame_id=str(i), timestamp=float(i), rgb_path="x.png", depth_path="x_depth.png", pose=pose))
    return frames


def test_write_rosbag_creates_readable_bag(tmp_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.default_rng(0).random((20, 3)))
    frames = _frames()

    bag_path = write_rosbag(pcd, frames, tmp_path)
    assert bag_path.exists()

    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    topics_seen = set()
    with Reader(bag_path) as reader:
        for connection, _timestamp, rawdata in reader.messages():
            topics_seen.add(connection.topic)
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            assert msg is not None

    assert "/spatial_os/points" in topics_seen
    assert "/spatial_os/path" in topics_seen
