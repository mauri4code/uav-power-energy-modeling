from pyulog import ULog
import pandas as pd

ulog = ULog("log_34_UnknownDate.ulg")

for d in ulog.data_list:
    if d.name in ["actuator_outputs", "actuator_motors"]:
        print("\nDATASET:", d.name)
        print("multi_id:", d.multi_id)
        print("fields:")
        for k in d.data.keys():
            print("  ", k)