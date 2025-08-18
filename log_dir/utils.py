#!/usr/bin/env python3
import os
import glob
import json
from pathlib import Path
from datetime import datetime

def get_latest_json_file(directory="."):
    """
    选择指定目录中日期最新的JSON文件

    Args:
        directory: 目录路径，默认为当前目录

    Returns:
        str: 最新JSON文件的路径，如果没有找到则返回None
    """

    # 方法1: 使用glob查找所有JSON文件
    json_pattern = os.path.join(directory, "*.json")
    json_files = glob.glob(json_pattern)

    if not json_files:
        print(f"在目录 '{directory}' 中没有找到JSON文件")
        return None

    # 获取文件修改时间并排序
    files_with_time = []
    for file_path in json_files:
        try:
            # 获取文件修改时间
            mtime = os.path.getmtime(file_path)
            files_with_time.append((file_path, mtime))
        except OSError as e:
            print(f"无法获取文件 {file_path} 的修改时间: {e}")
            continue

    if not files_with_time:
        print("没有可访问的JSON文件")
        return None

    # 按修改时间排序（最新的在前）
    files_with_time.sort(key=lambda x: x[1], reverse=True)

    latest_file = files_with_time[0][0]
    latest_time = datetime.fromtimestamp(files_with_time[0][1])

    print(f"找到 {len(json_files)} 个JSON文件")
    print(f"最新文件: {latest_file}")
    print(f"修改时间: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")

    return latest_file

def get_latest_json_file_pathlib(directory="."):
    """
    使用pathlib的替代实现
    """
    directory = Path(directory)

    # 查找所有JSON文件
    json_files = list(directory.glob("*.json"))

    if not json_files:
        print(f"在目录 '{directory}' 中没有找到JSON文件")
        return None

    # 按修改时间排序
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    latest_file = json_files[0]
    latest_time = datetime.fromtimestamp(latest_file.stat().st_mtime)

    print(f"找到 {len(json_files)} 个JSON文件")
    print(f"最新文件: {latest_file}")
    print(f"修改时间: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")

    return str(latest_file)

def list_all_json_files(directory="."):
    """
    列出目录中所有JSON文件及其修改时间
    """
    json_files = glob.glob(os.path.join(directory, "*.json"))

    if not json_files:
        print(f"在目录 '{directory}' 中没有找到JSON文件")
        return []

    files_info = []
    for file_path in json_files:
        try:
            mtime = os.path.getmtime(file_path)
            mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            files_info.append((file_path, mtime, mtime_str))
        except OSError:
            continue

    # 按时间排序
    files_info.sort(key=lambda x: x[1], reverse=True)

    print(f"\n所有JSON文件 (按时间排序):")
    print("-" * 60)
    for file_path, _, mtime_str in files_info:
        filename = os.path.basename(file_path)
        print(f"{filename:<30} {mtime_str}")

    return files_info

def load_latest_json(directory="."):
    """
    加载最新的JSON文件内容
    """
    latest_file = get_latest_json_file(directory)

    if latest_file is None:
        return None

    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"\n成功加载JSON文件，包含 {len(data)} 个元素" if isinstance(data, (list, dict)) else "\n成功加载JSON文件")
        return data
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return None
    except Exception as e:
        print(f"文件读取错误: {e}")
        return None

# 主函数示例
def main():
    """
    主函数 - 演示不同用法
    """
    print("=== 查找最新JSON文件 ===")

    # 1. 基本用法：获取最新JSON文件路径
    latest_json = get_latest_json_file(".")

    if latest_json:
        print(f"\n最新JSON文件: {latest_json}")

        # 2. 列出所有JSON文件
        print("\n" + "="*50)
        list_all_json_files(".")

        # 3. 加载最新JSON文件内容
        print("\n" + "="*50)
        data = load_latest_json(".")

        if data is not None:
            # 简单展示数据结构
            if isinstance(data, dict):
                print(f"JSON对象的键: {list(data.keys())}")
            elif isinstance(data, list):
                print(f"JSON数组长度: {len(data)}")
                if len(data) > 0:
                    print(f"第一个元素类型: {type(data[0])}")

    # 4. 指定其他目录
    # latest_in_logs = get_latest_json_file("./logs")

if __name__ == "__main__":
    main()


# 快速使用的便捷函数
def quick_get_latest_json():
    """一行代码获取最新JSON文件路径"""
    json_files = glob.glob("*.json")
    return max(json_files, key=os.path.getmtime) if json_files else None

# 使用示例：
# latest_file = quick_get_latest_json()
# if latest_file:
#     with open(latest_file, 'r') as f:
#         data = json.load(f)
#         print(f"加载了文件: {latest_file}")