import os
import glob

# 指定目标文件夹路径
folder_path = '/home/ks8018/dn-splatter/datasets/blackbunny3/depth'  # 替换为实际的文件夹路径

# 删除所有 .npy 文件
npy_files = glob.glob(os.path.join(folder_path, '*.npy'))
for npy_file in npy_files:
    os.remove(npy_file)
    print(f'Deleted: {npy_file}')

# # 查找并删除文件名为 c_i.png 且 i 不能整除 5 的文件
# png_files = glob.glob(os.path.join(folder_path, '*.png'))
# for png_file in png_files:
#     # 提取文件名中的 i
#     filename = os.path.basename(png_file)
#     i_str = filename.split('_')[1].split('.')[0]
#     if i_str.isdigit():
#         i = int(i_str)
#         if i % 5 != 0:
#             os.remove(png_file)
#             print(f'Deleted: {png_file}')

print("All specified files have been deleted.")