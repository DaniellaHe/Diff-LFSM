import json
import os
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import cv2

Train_or_Test = "Train"
# Train_or_Test = "Test"

# k = 5
k = 1
for i in range(k):
    ROOT_DIR = f"../MRI_vertebrae/{Train_or_Test}{i + 1}"
    output_dir = f"./{Train_or_Test}{i + 1}"
    # os.makedirs(output_dir, exist_ok=True)
    json_file = 'cropped_bbox.json'
    json_data_list = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            json_object = json.loads(line)
            json_data_list.append(json_object)
    label_list = ["S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10"]
    data_dict = []
    image_name = ['image', 'model=29525_t=150_gray_rec', 'color_sal_mask', 'color_mse', 'gradcam++']

    for name in os.listdir(ROOT_DIR):
        image_path = os.path.join(ROOT_DIR, name, "image.png")
        npy_name = f'ARGS=2_t=150_num={i+1}_saliency_map.npy'
        mask_path = os.path.join(ROOT_DIR, name, npy_name)
        if not os.path.exists(mask_path):
            continue
        mask = np.load(mask_path, allow_pickle=True)[0][0] * 255

        json_data = json_data_list[0][int(name.split('_')[-1])]
        cropped_bbox_list = json_data['cropped_bbox']
        print(name)
        print(json_data)
        image = Image.open(image_path)
        img_width, img_height = image.size

        # 将 mask 调整为与图像相同的大小
        mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        mask = Image.fromarray((mask * 255).astype(np.uint8))

        j = 1
        for label, l, cropped_bbox in cropped_bbox_list:

            expand_len = 20
            cropped_bbox = [
                max(0, cropped_bbox[0] - expand_len),
                max(0, cropped_bbox[1] - expand_len),
                min(img_width, cropped_bbox[2] + expand_len),
                min(img_height, cropped_bbox[3] + expand_len)
            ]
            cropped_img = image.crop(cropped_bbox)
            cropped_mask = mask.crop(cropped_bbox)

            file_name = (f"{name.split('_')[-1]}_{j}_{l}_{'unhealthy' if label == 1 else 'healthy'}")
            try:
                os.makedirs(os.path.join(output_dir, file_name))
            except OSError:
                pass
            cropped_img.save(os.path.join(output_dir, file_name, 'image.png'))
            cropped_mask.save(os.path.join(output_dir, file_name, f'saliency_map.png'))
            j = j + 1
            # print(image_path)
