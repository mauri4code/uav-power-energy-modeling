import os
import csv
from pathlib import Path
from collections import defaultdict

from rosbags.rosbag1 import Reader
from rosbags.typesys import get_types_from_msg, Stores, get_typestore

BAG_FILE = "uav_101_2026-04-01-19-08-46.bag"
OUTPUT_FOLDER = "uav_exported"

# IMPORTANT: use only REAL TOPIC NAMES here

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

TOPICS_TO_EXPORT = list(TOPIC_FIELDS.keys())

def flatten(obj, prefix=""):
    """Recursively flatten a rosbags message into a flat dict."""
    fields = {}

    if hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            value = getattr(obj, field)
            key = f"{prefix}.{field}" if prefix else field
            fields.update(flatten(value, key))

    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            key = f"{prefix}[{i}]"
            fields.update(flatten(item, key))

    else:
        fields[prefix] = obj

    return fields


def sanitize_topic_name(topic):
    return topic.strip("/").replace("/", "_")


def list_topics(bag_file):
    print(f"\nOpening bag: {bag_file}\n")

    with Reader(Path(bag_file)) as bag:
        print(f"{'Topic':<60} {'Type':<40} {'Messages'}")
        print("-" * 120)
        for topic, info in sorted(bag.topics.items()):
            print(f"{topic:<60} {info.msgtype:<40} {info.msgcount}")

    print()
def build_typestore_with_custom_msgs(bag, topics):
    typestore = get_typestore(Stores.ROS1_NOETIC)
    added_types = {}

    for connection in bag.connections:
        if connection.topic not in topics:
            continue

        if connection.msgtype in typestore.types:
            continue

        if hasattr(connection, "msgdef") and connection.msgdef:
            msgdef = connection.msgdef

            # if msgdef is a wrapper object, extract the text
            if hasattr(msgdef, "data"):
                msgdef = msgdef.data

            try:
                new_types = get_types_from_msg(msgdef, connection.msgtype)
                added_types.update(new_types)
            except Exception as e:
                print(f"Could not register {connection.msgtype}: {e}")

    if added_types:
        typestore.register(added_types)

    return typestore

def export_topics_to_csv(bag_file, output_folder, topics):
    os.makedirs(output_folder, exist_ok=True)
    typestore = None

    print(f"\nOpening bag file: {bag_file}")

    with Reader(Path(bag_file)) as bag:
        available_topics = list(bag.topics.keys())

        print(f"\nFound {len(available_topics)} topics in bag.")

        if not topics:
            topics = available_topics
            print("No topics specified, exporting ALL topics.")
        else:
            print("\nRequested topics:")
            for t in topics:
                print(f"  - {t}")

            missing = [t for t in topics if t not in available_topics]
            if missing:
                print("\nWarning: these topics were not found:")
                for m in missing:
                    print(f"  - {m}")

            topics = [t for t in topics if t in available_topics]

        if not topics:
            print("\nNo valid topics to export. Nothing will be written.")
            return
        typestore = build_typestore_with_custom_msgs(bag, topics)
        print("\nValid topics to export:")
        for t in topics:
            print(f"  - {t}")

        topic_rows = defaultdict(list)
        topic_headers = {}

        connections = [c for c in bag.connections if c.topic in topics]
        print(f"\nReading from {len(connections)} matching connections...")

        msg_counter = 0

        for connection, timestamp, rawdata in bag.messages(connections=connections):
            try:
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                data = flatten(msg)

                topic = connection.topic
                flat = {"timestamp": timestamp * 1e-9}

                for field in TOPIC_FIELDS[topic]:
                    colname = topic.strip("/").replace("/", "_") + "__" + field.replace(".", "_")
                    flat[colname] = data.get(field)

                topic_rows[topic].append(flat)

                if topic not in topic_headers:
                    topic_headers[topic] = list(flat.keys())
                else:
                    for key in flat.keys():
                        if key not in topic_headers[topic]:
                            topic_headers[topic].append(key)

                msg_counter += 1

            except Exception as e:
                print(f"Failed to deserialize message from {connection.topic}: {e}")

        print(f"\nRead {msg_counter} messages total.")

        print("\nWriting CSV files...")
        for topic in topics:
            rows = topic_rows.get(topic, [])
            if not rows:
                print(f"  - {topic}: no rows, skipped.")
                continue

            filename = sanitize_topic_name(topic) + ".csv"
            filepath = os.path.join(output_folder, filename)

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=topic_headers[topic],
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)

            print(f"  - Wrote {filepath} ({len(rows)} rows)")

    print(f"\nDone. CSV files are in:\n{os.path.abspath(output_folder)}\n")


if __name__ == "__main__":
    list_topics(BAG_FILE)
    export_topics_to_csv(BAG_FILE, OUTPUT_FOLDER, TOPICS_TO_EXPORT)