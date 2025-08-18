import json
from utils import get_latest_json_file_pathlib

# 读取 trace 文件
with open(get_latest_json_file_pathlib(), "r") as f:
    trace = json.load(f)

# 提取所有 GPU/CPU kernel 事件
events = []
for e in trace["traceEvents"]:
    if "dur" in e and "name" in e:
        # dur 是纳秒（ns）
        dur_ms = e["dur"]
        events.append((e["name"], dur_ms))

# 按耗时排序
events.sort(key=lambda x: x[1], reverse=True)

# 打印前 30
print("Top 10 ops by duration:")
for name, dur in events[:30]:
    name = name[:39] + "..." if len(name) > 39 else name
    print(f"{name:<50} {dur:.3f} us")
