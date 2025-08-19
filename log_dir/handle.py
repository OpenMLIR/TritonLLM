import json
from utils import get_latest_json_file_pathlib

# Read the latest trace JSON file
with open(get_latest_json_file_pathlib(), "r") as f:
    trace = json.load(f)

# Extract all GPU/CPU kernel events
events = []
for e in trace["traceEvents"]:
    if "dur" in e and "name" in e:
        # 'dur' is in nanoseconds (ns)
        dur_ms = e["dur"]
        events.append((e["name"], dur_ms))

# Sort events by duration in descending order
events.sort(key=lambda x: x[1], reverse=True)

print("Top 50~80 ops by duration:")
for name, dur in events[50:80]:
    # Truncate long names for readability
    name = name[:50] + "..." if len(name) > 50 else name
    print(f"{name:<55} {dur:.3f} us")
