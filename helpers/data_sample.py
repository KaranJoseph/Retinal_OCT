import os
import random
import shutil

orig_dir = 'D:\Github\Federated_Learning\Experiments\RetinalOCT\Data\\train'
new_dir = 'D:\Github\Federated_Learning\Retinal_OCT\data\\train'

classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

for cls in classes:
    files = os.listdir(os.path.join(orig_dir, cls))
    num_files = len(files)
    percent = 0.3
    num_select = int(num_files*percent)

    #Choose random files from directory
    selected_files = random.sample(files, num_select)

    #Create new directory
    os.mkdir(os.path.join(new_dir, cls))

    for file in selected_files:
        file_path = os.path.join(os.path.join(orig_dir, cls), file)
        new_path = os.path.join(os.path.join(new_dir, cls), file)
        shutil.copyfile(file_path, new_path)

    assert len(os.listdir(os.path.join(new_dir, cls)))>0
