import shutil
import json
import os

if os.path.isdir('dataset_classified_2'):
    shutil.rmtree('dataset_classified_2')
os.mkdir('dataset_classified_2')

for id in os.listdir('dataset'):
    source_directory = os.path.join('dataset', id)
    meta = json.load(open(os.path.join(source_directory, 'meta.json'), 'r'))
    category = meta['model_cat']
    category_directory = os.path.join('dataset_classified_2', category)
    if not os.path.isdir(category_directory):
        os.makedirs(category_directory)
    target_directory = os.path.join(category_directory, id)
    if os.path.isdir(os.path.join(source_directory, 'point_sample')):
        shutil.copytree(source_directory, target_directory)

