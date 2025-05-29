import pandas as pd
import glob
import os
import re
from datetime import datetime, timedelta

# 定义输入和输出目录
input_dir = 'data_all_processed_inter2' # 替换为你的输入文件目录
output_dir = 'data_all_label' # 替换为你的输出目录
vortex_centers_input = 'vortex_centers.csv'  # 涡旋中心信息文件
vortex_centers_output = os.path.join(output_dir, 'vortex_centers.csv')  # 输出涡旋中心文件

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取涡旋中心信息
try:
    vortex_centers_df = pd.read_csv(vortex_centers_input)
    # 确保日期列为整数
    vortex_centers_df['Year'] = vortex_centers_df['Year'].astype(int)
    vortex_centers_df['Month'] = vortex_centers_df['Month'].astype(int)
    vortex_centers_df['Day'] = vortex_centers_df['Day'].astype(int)
    # 创建 datetime 列
    vortex_centers_df['vortex_date'] = pd.to_datetime(vortex_centers_df[['Year', 'Month', 'Day']])
except Exception as e:
    print(f"Error reading vortex centers file {vortex_centers_input}: {e}")
    exit(1)

# 获取涡旋中心日期及其前后一天
def get_vortex_dates(vortex_date):
    return [
        vortex_date - timedelta(days=1),
        vortex_date,
        vortex_date + timedelta(days=1)
    ]

# 获取所有 itp*_深度学习.txt 文件
file_list = glob.glob(os.path.join(input_dir, 'itp*深度学习.txt'))

for file_path in file_list:
    print(f"Processing file: {file_path}")
    
    # 提取 ITP 编号
    itp_match = re.match(r'itp(\d+)深度学习\.txt', os.path.basename(file_path))
    if not itp_match:
        print(f"Warning: Filename {file_path} does not match expected pattern. Skipping.")
        continue
    itp = f"ITP{itp_match.group(1)}"
    
    # 查找对应 ITP 的所有涡旋中心
    vortex_info = vortex_centers_df[vortex_centers_df['Ipt'] == itp]
    if vortex_info.empty:
        print(f"Warning: No vortex center found for {itp} in {vortex_centers_input}. Skipping.")
        continue
    
    # 获取所有涡旋日期范围
    vortex_date_tuples = []
    for _, row in vortex_info.iterrows():
        vortex_dates = get_vortex_dates(row['vortex_date'])
        vortex_date_tuples.extend([(d.year, d.month, d.day) for d in vortex_dates])
    # 去重日期
    vortex_date_tuples = list(set(vortex_date_tuples))
    
    # 读取数据文件
    try:
        df = pd.read_csv(file_path, sep='\s+', header=0)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}. Skipping.")
        continue
    
    # 验证列数（预期 496 列）
    if len(df.columns) != 496:
        print(f"Warning: File {file_path} has unexpected column count: {len(df.columns)}. Skipping.")
        continue
    
    # 确保 Year, Month, Day 为整数
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['Day'] = df['Day'].astype(int)
    
    # 添加标签列
    df['Label'] = 0
    for year, month, day in vortex_date_tuples:
        df.loc[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day), 'Label'] = 1
    
    # 获取输出文件名
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    
    # 保存标签文件（空格分隔，保留表头）
    df.to_csv(output_file, sep=' ', index=False, na_rep='NaN', header=True)
    print(f"Saved labeled file: {output_file}")

# 复制涡旋中心信息到输出目录
vortex_centers_df[['Ipt', 'Year', 'Month', 'Day', 'Lon', 'Lat']].to_csv(vortex_centers_output, index=False)
print(f"Copied vortex centers to: {vortex_centers_output}")