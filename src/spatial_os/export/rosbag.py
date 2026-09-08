"""ROS bag export via the pure-Python `rosbags` library -- no ROS install required.

Writes sensor_msgs/PointCloud2 (the reconstructed cloud) and nav_msgs/Path
(the camera trajectory) into a single rosbag2 bag.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from spatial_os.schema import Frame
from spatial_os.utils.geometry import rotation_matrix_to_quaternion

FRAME_ID = "map"


def _pointcloud2_msg(typestore, points: np.ndarray, stamp_sec: int, stamp_nanosec: int):
    PointField = typestore.types["sensor_msgs/msg/PointField"]
    PointCloud2 = typestore.types["sensor_msgs/msg/PointCloud2"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    data = np.frombuffer(points.astype(np.float32).tobytes(), dtype=np.uint8)

    header = Header(stamp=Time(sec=stamp_sec, nanosec=stamp_nanosec), frame_id=FRAME_ID)
    return PointCloud2(
        header=header,
        height=1,
        width=len(points),
        fields=fields,
        is_bigendian=False,
        point_step=12,
        row_step=12 * len(points),
        data=data,
        is_dense=True,
    )


def _path_msg(typestore, frames: list[Frame], stamp_sec: int, stamp_nanosec: int):
    Path = typestore.types["nav_msgs/msg/Path"]
    PoseStamped = typestore.types["geometry_msgs/msg/PoseStamped"]
    Pose = typestore.types["geometry_msgs/msg/Pose"]
    Point = typestore.types["geometry_msgs/msg/Point"]
    Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]

    poses = []
    for frame in frames:
        m = frame.pose_matrix()
        t = m[:3, 3]
        q = rotation_matrix_to_quaternion(m[:3, :3])
        header = Header(stamp=Time(sec=stamp_sec, nanosec=stamp_nanosec), frame_id=FRAME_ID)
        pose = Pose(
            position=Point(x=float(t[0]), y=float(t[1]), z=float(t[2])),
            orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
        )
        poses.append(PoseStamped(header=header, pose=pose))

    header = Header(stamp=Time(sec=stamp_sec, nanosec=stamp_nanosec), frame_id=FRAME_ID)
    return Path(header=header, poses=poses)


def write_rosbag(pcd: o3d.geometry.PointCloud, frames: list[Frame], out_dir: str | Path, name: str = "session") -> Path:
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    bag_path = Path(out_dir) / f"{name}_rosbag"
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    points = np.asarray(pcd.points)
    stamp_sec, stamp_nanosec = 0, 0

    with Writer(bag_path, version=9) as writer:
        pc_conn = writer.add_connection(
            "/spatial_os/points", "sensor_msgs/msg/PointCloud2", typestore=typestore
        )
        path_conn = writer.add_connection("/spatial_os/path", "nav_msgs/msg/Path", typestore=typestore)

        pc_msg = _pointcloud2_msg(typestore, points, stamp_sec, stamp_nanosec)
        writer.write(pc_conn, 0, typestore.serialize_cdr(pc_msg, pc_conn.msgtype))

        path_msg = _path_msg(typestore, frames, stamp_sec, stamp_nanosec)
        writer.write(path_conn, 0, typestore.serialize_cdr(path_msg, path_conn.msgtype))

    return bag_path
