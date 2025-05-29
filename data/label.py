import pandas as pd
import glob
import os
from datetime import datetime, timedelta
import re

# 定义输入和输出目录
input_dir = 'ITP_data_processed_inter2' # 替换为你的输入文件目录
output_dir = 'ITP_data_label' # 替换为你的输出文件目录
vortex_centers_file = os.path.join(output_dir, 'vortex_centers.csv')  # 涡旋中心汇总文件

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 从文件名提取涡旋信息
def extract_vortex_info(filename):
    try:
        # 假设文件名格式为 itp{}_YYYY_MM_DD_LON_LAT.txt
        # 使用正则表达式提取信息
        pattern = r'itp(\d+)_(\d{4})_(\d{1,2})_(\d{1,2})_([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+).txt'
        match = re.match(pattern, os.path.basename(filename))
        if not match:
            print(f"Warning: Filename {filename} does not match expected pattern. Skipping.")
            return None
        itp = f"ITP{match.group(1)}"
        year = int(match.group(2))
        month = int(match.group(3))
        day = int(match.group(4))
        lon = float(match.group(5))
        lat = float(match.group(6))
        vortex_date = datetime(year, month, day)
        return {'itp': itp, 'year': year, 'month': month, 'day': day, 'lon': lon, 'lat': lat, 'vortex_date': vortex_date}
    except Exception as e:
        print(f"Warning: Cannot parse filename {filename}. Error: {e}. Skipping.")
        return None

# 获取涡旋中心日期及其前后一天
def get_vortex_dates(vortex_date):
    return [
        vortex_date - timedelta(days=1),
        vortex_date,
        vortex_date + timedelta(days=1)
    ]

# 初始化涡旋中心列表
vortex_centers = []

# 获取所有 .txt 文件（插值后文件）
file_list = glob.glob(os.path.join(input_dir, '*.txt'))

for file_path in file_list:
    print(f"Processing file: {file_path}")
    
    # 提取涡旋信息
    vortex_info = extract_vortex_info(file_path)
    if vortex_info is None:
        continue
    
    # 保存涡旋中心信息
    vortex_centers.append({
        'Ipt': vortex_info['itp'],
        'Year': vortex_info['year'],
        'Month': vortex_info['month'],
        'Day': vortex_info['day'],
        'Lon': vortex_info['lon'],
        'Lat': vortex_info['lat']
    })
    
    # 获取涡旋日期范围
    vortex_dates = get_vortex_dates(vortex_info['vortex_date'])
    vortex_date_tuples = [(d.year, d.month, d.day) for d in vortex_dates]
    
    # 读取文件（空格分隔，包含表头）
    df = pd.read_csv(file_path, sep='\s+', header=0)
    
    # 验证列数（预期 496 列）
    if len(df.columns) != 496:
        print(f"Warning: File {file_path} has unexpected column count: {len(df.columns)}. Skipping.")
        continue
    
    # 确保 Year, Month, Day 为整数
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['Day'] = df['Day'].astype(int)
    
    # 添加标签列
    df['Label'] = 0  # 默认标签为 0
    for year, month, day in vortex_date_tuples:
        df.loc[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == day), 'Label'] = 1
    
    # 获取输出文件名
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    
    # 保存标签文件（空格分隔，保留表头）
    df.to_csv(output_file, sep=' ', index=False, na_rep='NaN', header=True)
    print(f"Saved labeled file: {output_file}")

# 保存涡旋中心信息到 CSV
if vortex_centers:
    vortex_centers_df = pd.DataFrame(vortex_centers)
    vortex_centers_df.to_csv(vortex_centers_file, index=False)
    print(f"Saved vortex centers to: {vortex_centers_file}")
else:
    print("No valid vortex centers extracted.")