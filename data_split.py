import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def split_data_sliding_window(input_file, output_dir, time_steps_per_sample=24, stride=1):
    """
    以滑动窗口方式将包含多个时间步的单个txt文件拆分为多个样本，
    每个样本包含指定时间步数的数据，保存到单独的文件夹中
    
    参数:
    input_file: 输入的txt文件路径
    output_dir: 输出文件夹路径
    time_steps_per_sample: 每个样本包含的时间步数（默认为24）
    stride: 滑动窗口的步长（默认为1）
    """
    try:
        # 读取输入文件
        df = pd.read_csv(input_file, delim_whitespace=True, header=0, na_values=['NaN', 'nan'])
        
        # 验证必要列
        required_cols = ['station', 'year', 'month', 'day', 'lon', 'lat', 'depth', 'temperature', 'salinity']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"输入文件缺少列: {missing_cols}")
            return False
        
        # 转换为适当的数据类型
        for col in ['lon', 'lat', 'depth', 'temperature', 'salinity']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['station'] = df['station'].astype(str)
        
        # 按时间和位置分组
        groups = df.groupby(['station', 'year', 'month', 'day', 'lon', 'lat'])
        
        # 收集所有时间步数据
        time_step_data = []
        metadata = []
        for name, group in groups:
            group = group.sort_values(by='depth')
            time_step_data.append(group)
            metadata.append({
                'station': name[0],
                'year': name[1],
                'month': name[2],
                'day': name[3],
                'lon': name[4],
                'lat': name[5]
            })
        
        # 检查是否有足够的时间步
        total_time_steps = len(time_step_data)
        if total_time_steps < time_steps_per_sample:
            print(f"输入文件只有 {total_time_steps} 个时间步，小于所需 {time_steps_per_sample} 个时间步")
            return False
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 使用滑动窗口生成样本
        sample_count = 0
        for start_idx in range(0, total_time_steps - time_steps_per_sample + 1, stride):
            # 创建样本文件夹
            sample_dir = os.path.join(output_dir, f"sample_{sample_count:04d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # 保存当前窗口的每个时间步
            for i, idx in enumerate(range(start_idx, start_idx + time_steps_per_sample)):
                output_file = os.path.join(sample_dir, f"time_step_{i:02d}.txt")
                time_step_data[idx].to_csv(output_file, sep=' ', index=False, float_format='%.6f')
            
            print(f"已保存样本 {sample_count} 到 {sample_dir} (时间步 {start_idx} 到 {start_idx + time_steps_per_sample - 1})")
            sample_count += 1
        
        print(f"总共生成 {sample_count} 个样本")
        return True
    
    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {str(e)}")
        return False

if __name__ == '__main__':
    input_file = "test_data/test_data.txt"  # 输入文件路径
    output_dir = "test_data/split_samples"  # 输出目录路径
    time_steps_per_sample = 24  # 每个样本的时间步数
    stride = 1  # 滑动窗口步长
    
    success = split_data_sliding_window(input_file, output_dir, time_steps_per_sample, stride)
    if success:
        print("数据拆分完成")
    else:
        print("数据拆分失败")