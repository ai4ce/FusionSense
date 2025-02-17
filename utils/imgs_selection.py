import os
import shutil
import json

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--path', type=str, required=True, help='input your path')
# args = parser.parse_args()
# path = args.path

def select_imgs(path, output_dir):
    source_folder = f'{path}/images'
    destination_folder = f'{output_dir}/selected_images'
    indices_file = f'{path}/train.txt'

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    with open(indices_file, 'r') as f:
        indices_line = f.readline().strip()
        target_indices = list(map(int, indices_line.split(',')))

    # 遍历目标索引
    for i in target_indices:
        file_name = f'rgb_{i}.png'
        source_file_path = os.path.join(source_folder, file_name)
        
        # 检查文件是否存在
        if os.path.exists(source_file_path):
            destination_file_path = os.path.join(destination_folder, file_name)
            shutil.copy(source_file_path, destination_file_path)
            print(f'{file_name} copy {destination_folder}')
        else:
            print(f'{file_name} not found in {source_folder}')

def filter_transform_json(path):
    indices_file = f'{path}/train.txt'
    transform_file = f'{path}/transforms.json'
    output_file = f'{path}/transforms.json'

    # 读取选中的图片ID
    with open(indices_file, 'r') as f:
        indices_line = f.readline().strip()
        target_indices = list(map(int, indices_line.split(',')))

    # 读取原始transform数据
    with open(transform_file, 'r') as f:
        transform_data = json.load(f)

    # 筛选出符合条件的frames
    filtered_frames = []
    for frame in transform_data.get('frames', []):
        file_path = frame['file_path']
        file_index = int(os.path.splitext(os.path.basename(file_path))[0].split('_')[1])
        
        if file_index in target_indices:
            filtered_frames.append(frame)

    train_filenames = [frame['file_path'] for frame in filtered_frames]
    all_file_paths = {frame['file_path'] for frame in transform_data.get('frames', [])}
    test_filenames = list(all_file_paths - set(train_filenames))

    transform_data['train_filenames'] = train_filenames
    transform_data['test_filenames'] = test_filenames
    transform_data['val_filenames'] = test_filenames[:10]
    # transform_data['frames'] = filtered_frames

    with open(output_file, 'w') as f:
        json.dump(transform_data, f, indent=4)

    print(f'Filtered transform data saved to {output_file}')

# if __name__ == "__main__":
#     select_imgs(path)
#     filter_transform_json(path)