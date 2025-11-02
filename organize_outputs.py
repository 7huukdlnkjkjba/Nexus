#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
整理测试输出文件

将所有test_output_*文件夹和nexus_simulation_*.log文件移动到一个集中的输出目录
"""

import os
import shutil
import glob
import datetime

# 创建集中存放的目录
output_root = "simulation_outputs"
if not os.path.exists(output_root):
    os.makedirs(output_root)
    print(f"创建集中输出目录: {output_root}")
else:
    print(f"使用已存在的输出目录: {output_root}")

# 移动test_output_*文件夹
test_output_dirs = glob.glob("test_output_*")
moved_dirs = 0
for dir_path in test_output_dirs:
    if os.path.isdir(dir_path):
        target_path = os.path.join(output_root, dir_path)
        # 避免覆盖，添加时间戳后缀
        if os.path.exists(target_path):
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            target_path = os.path.join(output_root, f"{dir_path}_{timestamp}")
        
        shutil.move(dir_path, target_path)
        moved_dirs += 1
        print(f"移动文件夹: {dir_path} -> {target_path}")

# 移动nexus_simulation_*.log文件
log_files = glob.glob("nexus_simulation_*.log")
moved_logs = 0
for log_file in log_files:
    if os.path.isfile(log_file):
        target_path = os.path.join(output_root, log_file)
        # 避免覆盖，添加时间戳后缀
        if os.path.exists(target_path):
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            name, ext = os.path.splitext(log_file)
            target_path = os.path.join(output_root, f"{name}_{timestamp}{ext}")
        
        shutil.move(log_file, target_path)
        moved_logs += 1
        print(f"移动日志文件: {log_file} -> {target_path}")

print(f"\n整理完成！")
print(f"共移动了 {moved_dirs} 个测试输出文件夹")
print(f"共移动了 {moved_logs} 个模拟日志文件")
print(f"所有文件已集中到: {output_root}")