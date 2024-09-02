# 1. Convert JSON to PNG, save to the patient folder (full image)
# 2. Divide the patient folder (full image) into Train and Test folders (4:1 ratio)
# 3. Generate cropped_bbox.json, which contains the names and positions of each vertebrae bounding box

import json
import os
import random
import shutil

import numpy as np
from PIL import Image

with open("/data/hsd/VSDiff/data/VerTumor600/MRI_vertebrae/detection_ver200204.json", "r") as f:
    # with open("./detection_ver200204.json", "r") as f:
    s = json.load(f)
images = s['images']
images = np.array(images)
masks = s['masks']
masks = np.array(masks)
cancer_diag = s['cancer_diag']
cancer_diag_sus = s['cancer_diag_sus']
all_patients = images.shape[0]
cancer = []

for p in range(all_patients):
    temp_cancer = []
    temp1 = cancer_diag[p]
    temp2 = cancer_diag_sus[p]
    for i in temp1:
        temp_cancer.append(i)
    for j in temp2:
        if j not in temp_cancer:
            temp_cancer.append(j)
    if len(temp_cancer) > 1 and '' in temp_cancer:
        temp_cancer.remove('')
    cancer.append(temp_cancer)


def get_id_list(cancer_temp):
    id_list = []
    classes = ["S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10"]
    for temp in cancer_temp:
        if temp == 'S':
            temp = 'S1'
        id_list.append(classes.index(temp) + 1)
    return id_list


h_num = 0
unh_num = 0
data_dict = []
# Train_or_Test = "Train"
label_list = ["S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10"]

# #############################
# 1. Save JSON to the patient folder
# The following for loop saves each patient's PNG images to the ./patient folder
# patient/healthy0/image.png
# patient/healthy0/mask.png
# patient/healthy0/unhealthy_mask.png
# patient/unhealthy1/image.png
# patient/unhealthy1/mask.png
# patient/unhealthy1/unhealthy_mask.png
# #############################
for idx in range(all_patients):
    img = images[idx].astype(np.uint8)
    mask = masks[idx].astype(np.uint8)
    cancer_tmp = cancer[idx]

    healthy_mask = mask.copy()
    unhealthy_mask = np.zeros_like(mask)
    save_image = np.zeros_like(mask)
    square_mask = np.zeros_like(mask)
    mask[mask > 0] = 255

    if cancer[idx][0] == '':
        save_dir = "./patient/healthy_{}/".format(idx)
        try:
            os.makedirs(save_dir)
        except OSError:
            pass

        img = Image.fromarray(img)
        img.save(save_dir + "image.png")

        msk = Image.fromarray(mask)
        msk.save(save_dir + "mask.png")

        unmsk = Image.fromarray(unhealthy_mask)
        unmsk.save(save_dir + "unhealthy_mask.png")

        h_num = h_num + 1

    elif cancer[idx][0] != '':

        save_dir = "./patient/unhealthy_{}/".format(idx)
        try:
            os.makedirs(save_dir)
        except OSError:
            pass

        img = Image.fromarray(img)
        img.save(save_dir + "image.png")

        msk = Image.fromarray(mask)
        msk.save(save_dir + "mask.png")

        cancer_temp = cancer[idx]
        unhealthy_id_list = get_id_list(cancer_temp)

        for un_id in unhealthy_id_list:
            unhealthy_mask[healthy_mask == un_id] = 255

        unhealthy_mask = unhealthy_mask.astype(np.uint8)
        unhealthy_mask = Image.fromarray(unhealthy_mask)
        unhealthy_mask.save(save_dir + "unhealthy_mask.png")

        unh_num = unh_num + 1

    if idx == all_patients - 1:
        print(h_num)
        print(unh_num)

# #############################
# 2. Divide the patient folder (full image) into Train and Test folders (4:1 ratio)
# #############################
data_folder = './patient'
healthy_folders = [folder for folder in os.listdir(data_folder) if folder.startswith('healthy')]
unhealthy_folders = [folder for folder in os.listdir(data_folder) if folder.startswith('unhealthy')]
random.seed(42)
random.shuffle(healthy_folders)
random.shuffle(unhealthy_folders)

k = 5
healthy_folds = [healthy_folders[i::k] for i in range(k)]
unhealthy_folds = [unhealthy_folders[i::k] for i in range(k)]


def safe_copytree(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


for i in range(k):
    train_folder = f'./Train{i + 1}'
    test_folder = f'./Test{i + 1}'
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    healthy_test = healthy_folds[i]
    unhealthy_test = unhealthy_folds[i]
    healthy_train = [folder for s in healthy_folds[:i] + healthy_folds[i + 1:] for folder in s]
    unhealthy_train = [folder for s in unhealthy_folds[:i] + unhealthy_folds[i + 1:] for folder in s]

    for folder in healthy_train:
        src_path = os.path.join(data_folder, folder)
        dst_path = os.path.join(train_folder, folder)
        safe_copytree(src_path, dst_path)

    for folder in healthy_test:
        src_path = os.path.join(data_folder, folder)
        dst_path = os.path.join(test_folder, folder)
        safe_copytree(src_path, dst_path)

    for folder in unhealthy_train:
        src_path = os.path.join(data_folder, folder)
        dst_path = os.path.join(train_folder, folder)
        safe_copytree(src_path, dst_path)

    for folder in unhealthy_test:
        src_path = os.path.join(data_folder, folder)
        dst_path = os.path.join(test_folder, folder)
        safe_copytree(src_path, dst_path)

# #############################
# 3. Generate cropped_bbox.json, which contains the names and positions of each vertebrae bounding box
# #############################
for idx in range(all_patients):
    img = images[idx].astype(np.uint8)
    mask = masks[idx].astype(np.uint8)
    labels = np.unique(mask)[1:]
    label_and_cropped_bbox_list = []
    cropped_bbox_list = []
    for label in labels:
        object_mask = (mask == label)
        coords = np.column_stack(np.where(object_mask))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped_bbox = (x_min, y_min, x_max + 1, y_max + 1)
        label_and_cropped_bbox = (label, cropped_bbox)
        # print(label_and_cropped_bbox)
        label_and_cropped_bbox_list.append(label_and_cropped_bbox)

    label_and_cropped_bbox_list.sort(key=lambda item: item[1][1])
    # print(label_and_cropped_bbox_list)

    for i in range(len(cancer[idx])):
        if cancer[idx][i] == 'S':
            cancer[idx][i] = 'S1'

    j = 1
    for label, cropped_bbox in label_and_cropped_bbox_list:
        # cropped_img = img.crop(cropped_bbox)
        # cropped_rec = rec.crop(cropped_bbox)
        # cropped_out_mask = out_mask.crop(cropped_bbox)

        l = label_list[label - 1]
        if l in cancer[idx]:
            cropped_bbox_item = (1, l, cropped_bbox)
            # save_dir = output_dir + f"/{name}_{j}_{l}_unhealthy"
            label_value = 1
        else:
            cropped_bbox_item = (0, l, cropped_bbox)
            # save_dir = output_dir + f"/{name}_{j}_{l}_healthy"
            label_value = 0

        cropped_bbox_list.append(cropped_bbox_item)

        j = j + 1

    # print(cropped_bbox_list)
    cropped_bbox_dict = {'id': idx, 'cropped_bbox': cropped_bbox_list, 'size': (img.shape[0], img.shape[1])}
    print(cropped_bbox_dict)
    data_dict.append(cropped_bbox_dict)

for item in data_dict:
    item['cropped_bbox'] = [(label, l, tuple(map(int, box))) for label, l, box in item['cropped_bbox']]

sorted_data_dict = sorted(data_dict, key=lambda x: int(x['id']))
print(len(sorted_data_dict))
output_file = "../cropped_vertebrae/cropped_bbox.json"
with open(output_file, "w") as json_file:
    json.dump(sorted_data_dict, json_file)

print(f"Sorted JSON data has been saved to {output_file}")
