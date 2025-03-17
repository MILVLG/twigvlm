import os
import re

def extract_output_dirs(log_folder):
    # 正则表达式匹配 --output_dir 后面的值
    output_dir_pattern = re.compile(r'--output_dir\s+(\S+)')
    
    # 用于存储结果的列表
    results = []
    
    # 遍历文件夹中的所有 .log 文件
    for root, dirs, files in os.walk(log_folder):
        for file in files:
            if file.endswith('.log'):
                file_path = os.path.join(root, file)
                
                # 获取文件的修改时间
                file_mtime = os.path.getmtime(file_path)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    output_dirs = []

                    for line in lines:
                        matches = output_dir_pattern.findall(line)
                        output_dirs.extend(matches)
                    
                    # 将文件名、行数、--output_dir 值和修改时间添加到结果列表
                    results.append((file, len(lines), output_dirs, file_mtime))
    
    # 按照修改时间对结果进行排序
    results.sort(key=lambda x: x[3])
    
    # 打印排序后的结果
    for result in results:
        file, line_count, output_dirs, file_mtime = result
        print(f"{file}\t{line_count}\t{output_dirs}")

# 调用函数并指定日志文件夹路径
log_folder_path = 'logs'  # 请替换为实际的文件夹路径
extract_output_dirs(log_folder_path)