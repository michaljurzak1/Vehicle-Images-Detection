import os
import sys
import json
import yaml
import pandas as pd
import shutil

def change_file_structure(save_dir=None, source_dir=None, dest_dir=None, new_dir_structure=None, df_subdir_mapping=None):
    if save_dir is None: save_dir = "csv_data"

    if source_dir is None: source_dir = "archive/trafic_data"
    if dest_dir is None: dest_dir = 'dataset'

    if new_dir_structure is None: new_dir_structure = {
        dest_dir: {
            'images': {
                'train': None,
                'valid': None,
                'test': None
            },
            'labels': {
                'train': None,
                'valid': None,
                'test': None
            }
        }
    }

    if df_subdir_mapping is None: df_subdir_mapping = {
        'df_train': 'train',
        'df_valid': 'valid',
        'df_test': 'test'
    }

    def load_csv_data(csv_file):
        df = pd.read_csv(csv_file)
        return df

    df_train = load_csv_data(os.path.join(save_dir, "train.csv"))
    df_valid = load_csv_data(os.path.join(save_dir, "valid.csv"))
    df_test = load_csv_data(os.path.join(save_dir, "test.csv"))


    for root, subdirs in new_dir_structure.items():
        for subdir, subsubdirs in subdirs.items():
            for subsubdir in subsubdirs:
                os.makedirs(os.path.join(root, subdir, subsubdir), exist_ok=True)

    
    for df_name, subdir in df_subdir_mapping.items():
        df = locals()[df_name]
        for _, row in df.iterrows():
            old_image_path = os.path.join(row['directory'], row['image'])
            new_image_path = os.path.join('dataset', 'images', subdir, row['image'])
            shutil.copyfile(old_image_path, new_image_path)

            old_label_path = os.path.join(row['label_directory'], row['image'].replace('.jpg', '.txt'))
            new_label_path = os.path.join('dataset', 'labels', subdir, row['image'].replace('.jpg', '.txt'))
            shutil.copyfile(old_label_path, new_label_path)

    # Copy the yaml file
    file_name = 'data_1.yaml'

    old_file_path = os.path.join(source_dir, file_name)
    new_file_path = os.path.join(dest_dir, file_name)

    shutil.copyfile(old_file_path, new_file_path)

    with open(new_file_path, 'r') as file:
        lines = file.readlines()

    # Change these lines
    lines[0] = f'train: {os.path.abspath(os.path.join(dest_dir, "images/train"))}\n'
    lines[1] = f'val: {os.path.abspath(os.path.join(dest_dir, "images/valid"))}\n'

    with open(new_file_path, 'w') as file:
        file.writelines(lines)

    with open(".gitignore", "a+") as file:
        file.seek(0)
        lines = file.readlines()
        if "dataset/\n" not in lines:
            file.write("\ndataset/")