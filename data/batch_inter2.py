import pandas as pd
import numpy as np
import glob
import os


# 定义输入和输出目录
input_dir = 'data/data_all_processed_inter1'  # 替换为你的输入文件目录
output_dir = 'data_all_processed_inter2'  # 替换为你的输出文件目录

# 目标列数
TARGET_COLS = 496

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取所有 .txt 或 .csv 文件
file_list = glob.glob(os.path.join(input_dir, '*.txt'))

for file_path in file_list:
    print(f"Processing file: {file_path}")
    
    # 读取文件（假设第一行为表头，空格分隔）
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 提取表头
    header = lines[0].strip().split()
    data_rows = [line.strip().split() for line in lines[1:]]
    
    # 确定列数
    num_cols = len(header)
    
    # 验证列数（496 或 252）
    if num_cols not in [496, 252]:
        print(f"Warning: File {file_path} has unexpected column count: {num_cols}. Skipping.")
        continue
    
    # 如果是 252 列，补齐表头和数据到 496 列
    if num_cols == 252:
        # 补齐表头
        num_pairs = (num_cols - 6) // 2  # 当前成对数值数量（123 对）
        additional_pairs = (496 - num_cols) // 2  # 需要补齐的成对数值（122 对）
        header.extend([f'Val{i}_{j}' for i in range(num_pairs + 1, num_pairs + additional_pairs + 1) for j in [1, 2]])
        
        # 补齐数据行
        for row in data_rows:
            row.extend(['NaN'] * (496 - num_cols))  # 补齐 244 列 NaN
        num_cols = 496
    
    # 创建 DataFrame，保留原始表头
    df = pd.DataFrame(data_rows, columns=header)
    
    # 将数值列转换为浮点型（从第 4 列开始，索引 4 对应 Lon）
    for col in df.columns[4:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 按列对成对数值进行线性插值（从第 6 列开始，索引 6 对应 Val1_1）
    for col in df.columns[6:]:
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
    
    # 填充剩余的 NaN（如果插值后仍有 NaN，例如整列为 NaN）
    for col in df.columns[6:]:
        if df[col].isna().all():
            df[col].fillna(0, inplace=True)  # 用 0 填充全 NaN 列
        else:
            df[col].fillna(method='ffill', inplace=True)  # 前向填充
            df[col].fillna(method='bfill', inplace=True)  # 后向填充
    columns_to_keep = ['Track', 'Year', 'Month', 'Day', 'Lon', 'Lat'] + \
                  [f'Val{i}_{j}' for i in range(5, 144) for j in [1, 2]]

    filtered_data = df[columns_to_keep]
    # 获取输出文件名
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    
    # 保存结果（空格分隔，保留表头）
    df.to_csv(output_file, sep=' ', index=False, na_rep='NaN', header=True)
    print(f"Saved interpolated file: {output_file}")