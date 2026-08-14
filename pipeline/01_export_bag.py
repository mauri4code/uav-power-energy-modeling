"""
Step 1 – Export ROS bag topics to CSV files.

Reads BAG_FILE and writes one CSV per topic into OUTPUT_FOLDER.

Run standalone : python 01_export_bag.py
Run via runner : called automatically by run_flight.py
"""

import os
import csv
from pathlib import Path
from collections import defaultdict

from rosbags.rosbag1 import Reader
from rosbags.typesys import get_types_from_msg, Stores, get_typestore

# ---- Topics and fields to export ----
TOPIC_FIELDS = {
    "/uav1/estimation_manager/uav_state": [
        "velocity.linear.x",
        "velocity.linear.y",
        "velocity.linear.z",
    ],
    "/uav1/hw_api/imu": [
        "angular_velocity.x",
        "angular_velocity.y",
        "angular_velocity.z",
        "linear_acceleration.x",
        "linear_acceleration.y",
        "linear_acceleration.z",
    ],
    "/uav1/hw_api/orientation": [
        "quaternion.x",
        "quaternion.y",
        "quaternion.z",
        "quaternion.w",
    ],
    "/uav1/mavros/altitude": [
        "local",
    ],
    "/uav1/mrs_uav_status/uav_status": [
        "battery_curr",
        "battery_volt",
    ],
}


def flatten(obj, prefix=""):
    fields = {}
    if hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            value = getattr(obj, field)
            key = f"{prefix}.{field}" if prefix else field
            fields.update(flatten(value, key))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            fields.update(flatten(item, f"{prefix}[{i}]"))
    else:
        fields[prefix] = obj
    return fields


def sanitize(topic):
    return topic.strip("/").replace("/", "_")


def build_typestore(bag, topics):
    typestore = get_typestore(Stores.ROS1_NOETIC)
    added = {}
    for conn in bag.connections:
        if conn.topic not in topics or conn.msgtype in typestore.types:
            continue
        if hasattr(conn, "msgdef") and conn.msgdef:
            msgdef = conn.msgdef.data if hasattr(conn.msgdef, "data") else conn.msgdef
            try:
                added.update(get_types_from_msg(msgdef, conn.msgtype))
            except Exception as e:
                print(f"  Could not register {conn.msgtype}: {e}")
    if added:
        typestore.register(added)
    return typestore


def run(bag_file, output_folder):
    topics = list(TOPIC_FIELDS.keys())

    os.makedirs(output_folder, exist_ok=True)
    print(f"\n[Step 1] Exporting bag: {bag_file}")

    with Reader(Path(bag_file)) as bag:
        available = set(bag.topics.keys())
        missing = [t for t in topics if t not in available]
        if missing:
            print(f"  WARNING – topics not found: {missing}")
        topics = [t for t in topics if t in available]

        typestore  = build_typestore(bag, topics)
        topic_rows = defaultdict(list)
        topic_hdrs = {}
        connections = [c for c in bag.connections if c.topic in topics]

        for conn, timestamp, rawdata in bag.messages(connections=connections):
            try:
                msg  = typestore.deserialize_ros1(rawdata, conn.msgtype)
                data = flatten(msg)
                flat = {"timestamp": timestamp * 1e-9}
                for field in TOPIC_FIELDS[conn.topic]:
                    col = sanitize(conn.topic) + "__" + field.replace(".", "_")
                    flat[col] = data.get(field)
                topic_rows[conn.topic].append(flat)
                if conn.topic not in topic_hdrs:
                    topic_hdrs[conn.topic] = list(flat.keys())
            except Exception as e:
                print(f"  Deserialize error ({conn.topic}): {e}")

        for topic in topics:
            rows = topic_rows.get(topic, [])
            if not rows:
                print(f"  SKIP (no data): {topic}")
                continue
            filepath = os.path.join(output_folder, sanitize(topic) + ".csv")
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=topic_hdrs[topic], extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Wrote {filepath}  ({len(rows)} rows)")

    print(f"  Done → {os.path.abspath(output_folder)}\n")


if __name__ == "__main__":
    print("Run via: python run_flight.py <test_folder>")
