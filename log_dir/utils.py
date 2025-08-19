#!/usr/bin/env python3
import os
import glob
import json
from pathlib import Path
from datetime import datetime

def get_latest_json_file(directory="."):
    """
    Select the most recently modified JSON file in the given directory.

    Args:
        directory: Path to the directory (default is current directory)

    Returns:
        str: Path to the latest JSON file, or None if not found
    """

    # Method 1: use glob to find all JSON files
    json_pattern = os.path.join(directory, "*.json")
    json_files = glob.glob(json_pattern)

    if not json_files:
        print(f"No JSON files found in directory '{directory}'")
        return None

    # Get modification time and sort
    files_with_time = []
    for file_path in json_files:
        try:
            mtime = os.path.getmtime(file_path)
            files_with_time.append((file_path, mtime))
        except OSError as e:
            print(f"Cannot get modification time for {file_path}: {e}")
            continue

    if not files_with_time:
        print("No accessible JSON files")
        return None

    # Sort by modification time (latest first)
    files_with_time.sort(key=lambda x: x[1], reverse=True)

    latest_file = files_with_time[0][0]
    latest_time = datetime.fromtimestamp(files_with_time[0][1])

    print(f"Found {len(json_files)} JSON files")
    print(f"Latest file: {latest_file}")
    print(f"Modification time: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")

    return latest_file

def get_latest_json_file_pathlib(directory="."):
    """
    Alternative implementation using pathlib
    """
    directory = Path(directory)

    # Find all JSON files
    json_files = list(directory.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in directory '{directory}'")
        return None

    # Sort by modification time
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    latest_file = json_files[0]
    latest_time = datetime.fromtimestamp(latest_file.stat().st_mtime)

    print(f"Found {len(json_files)} JSON files")
    print(f"Latest file: {latest_file}")
    print(f"Modification time: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")

    return str(latest_file)

def list_all_json_files(directory="."):
    """
    List all JSON files in the directory with modification time
    """
    json_files = glob.glob(os.path.join(directory, "*.json"))

    if not json_files:
        print(f"No JSON files found in directory '{directory}'")
        return []

    files_info = []
    for file_path in json_files:
        try:
            mtime = os.path.getmtime(file_path)
            mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            files_info.append((file_path, mtime, mtime_str))
        except OSError:
            continue

    # Sort by modification time
    files_info.sort(key=lambda x: x[1], reverse=True)

    print(f"\nAll JSON files (sorted by modification time):")
    print("-" * 60)
    for file_path, _, mtime_str in files_info:
        filename = os.path.basename(file_path)
        print(f"{filename:<30} {mtime_str}")

    return files_info

def load_latest_json(directory="."):
    """
    Load the content of the latest JSON file
    """
    latest_file = get_latest_json_file(directory)

    if latest_file is None:
        return None

    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"\nSuccessfully loaded JSON file containing {len(data)} items" 
              if isinstance(data, (list, dict)) else "\nSuccessfully loaded JSON file")
        return data
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        print(f"File reading error: {e}")
        return None

# Example main function
def main():
    """
    Main function - demonstrates usage
    """
    print("=== Find the latest JSON file ===")

    # 1. Basic usage: get latest JSON file path
    latest_json = get_latest_json_file(".")

    if latest_json:
        print(f"\nLatest JSON file: {latest_json}")

        # 2. List all JSON files
        print("\n" + "="*50)
        list_all_json_files(".")

        # 3. Load content of latest JSON file
        print("\n" + "="*50)
        data = load_latest_json(".")

        if data is not None:
            # Show basic data structure
            if isinstance(data, dict):
                print(f"Keys in JSON object: {list(data.keys())}")
            elif isinstance(data, list):
                print(f"Length of JSON array: {len(data)}")
                if len(data) > 0:
                    print(f"Type of first element: {type(data[0])}")

    # 4. Specify another directory
    # latest_in_logs = get_latest_json_file("./logs")

if __name__ == "__main__":
    main()


# Convenience function for quick access
def quick_get_latest_json():
    """Get the latest JSON file path in one line"""
    json_files = glob.glob("*.json")
    return max(json_files, key=os.path.getmtime) if json_files else None

# Usage example:
# latest_file = quick_get_latest_json()
# if latest_file:
#     with open(latest_file, 'r') as f:
#         data = json.load(f)
#         print(f"Loaded file: {latest_file}")
