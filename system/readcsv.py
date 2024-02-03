import os
import pandas as pd

def read_csv_files_in_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                file_list.append(file_path)

    dataframes = []
    for file_path in file_list:
        df = pd.read_csv(file_path)
        dataframes.append(df)

    return dataframes

# 指定要读取的目录路径
directory_path = '/path/to/directory'

# 读取目录中的所有 CSV 文件
dfs = read_csv_files_in_directory(directory_path)

# 打印每个 DataFrame 的内容
for df in dfs:
    print(df.head())

import os
import pandas as pd

# 创建子文件夹
subdirectory = 'data_folder'
if not os.path.exists(subdirectory):
    os.makedirs(subdirectory)

# 生成示例 DataFrame
data = {'Name': ['John', 'Emma', 'Robert'],
        'Age': [28, 32, 45],
        'City': ['New York', 'London', 'Paris']}
df = pd.DataFrame(data)

# 保存 DataFrame 到子文件夹
filename = 'data.csv'
save_path = os.path.join(subdirectory, filename)
df.to_csv(save_path, index=False)

