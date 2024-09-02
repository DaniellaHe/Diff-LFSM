import json
import os
import random

import cv2
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from perlin import rand_perlin_2d_np
from torch.nn.utils.rnn import pad_sequence

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def spine_collate(batch):
    images = [item['images'] for item in batch]
    sals = [item['sals'] for item in batch]
    labels = [item['labels'] for item in batch]
    patient_id = [item['patient_id'] for item in batch]

    images_padded = pad_sequence([torch.stack(img) for img in images], batch_first=True, padding_value=0)
    sals_padded = pad_sequence([torch.stack(sal) for sal in sals], batch_first=True, padding_value=0)
    labels_padded = pad_sequence([torch.tensor(lbl) for lbl in labels], batch_first=True, padding_value=-1)
    # patient_id_padded = pad_sequence([torch.tensor(lbl) for lbl in patient_id], batch_first=True, padding_value=" ")
    patient_id_padded = pad_sequence(
        [torch.tensor(list(map(int, lbl))) for lbl in patient_id],
        batch_first=True,
        padding_value=-1
    )
    return {'images': images_padded,
            'sals': sals_padded,
            'labels': labels_padded,
            'patient_id': patient_id_padded}

def scale_array(array, new_min=0, new_max=255):
    min_val = np.min(array)
    max_val = np.max(array)
    scaled_array = ((array - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
    scaled_array = scaled_array.astype(np.uint8)

    return scaled_array


def get_freq_image(slice):
    val = random.randint(2, 10)
    fft_image = np.fft.fftshift(np.fft.fft2(slice))
    mask = np.zeros_like(fft_image)
    rows, cols = fft_image.shape
    center_row, center_col = rows // 2, cols // 2
    mask_range = range(center_row - val, center_row + val)
    choice = random.randint(0, 1)
    # if choice == 0:
    #     mask[mask_range, center_col - val: center_col + val] = 1
    # else:
    mask[mask_range, center_col - val: center_col + val] = 1
    mask = 1 - mask
    masked_fft = np.multiply(fft_image, mask)
    reconstructed_image = np.real(np.fft.ifft2(np.fft.ifftshift(masked_fft)))

    return reconstructed_image


def visualize_IPM_FPM_image(img, fft_image, healthy_mask, unhealthy_mask, id):
    scale_fft_image = scale_array(fft_image)
    scale_img = scale_array(img)
    rate = 0.5
    perlin_scale = 6
    min_perlin_scale = 0

    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_noise = rand_perlin_2d_np((img.shape[0], img.shape[1]), (perlin_scalex, perlin_scaley))

    rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
    perlin_noise = rot(image=perlin_noise)
    threshold = 0.5
    perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
    perlin_thr = torch.from_numpy(perlin_thr)

    msk = perlin_thr.float()
    msk = msk * healthy_mask
    msk = msk.numpy()
    beta = torch.rand(1).numpy()[0] * 0.2 + 0.6

    scale_masked_image = scale_fft_image * msk * beta + scale_img * msk * (1 - beta) + scale_img * (1 - msk)
    masked_image = scale_masked_image * (1 - unhealthy_mask)
    masked_image = masked_image.astype(np.float32)

    # vshow = True
    vshow = False
    if vshow:
        plt.figure(figsize=(5, 5))
        folder_path = f'./data/IPM_FPM_image/{id}/'
        os.makedirs(folder_path, exist_ok=True)

        pic = {
            'image': scale_img,
            'fft_image': scale_fft_image,
            'perlin_thr': perlin_thr.numpy(),
            'healthy_mask01': healthy_mask,
            'mask01': msk,
            'Imask01': 1 - msk,
            'unhealthy_mask01': unhealthy_mask,
            'Iunhealthy_mask01': 1 - unhealthy_mask,
            'masked_image': masked_image,
            'maskunhealthy_image': (1 - unhealthy_mask) * scale_img,
        }

        for title, img in pic.items():
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            img = img.astype(np.uint8)

            height, width = img.shape
            min_dim = min(height, width)
            cropped_img = img[:min_dim, :min_dim]
            if 'mask01' in title:
                cropped_img = (cropped_img * 255)

            image_to_save = Image.fromarray(cropped_img)
            image_to_save.save(folder_path + f'{title}.png')

            plt.close()

        data = {
            'image': scale_img,
            'fft_image': scale_fft_image,
            'perlin_thr': perlin_thr.numpy(),
            'healthy_mask': healthy_mask,
            'msk': msk,
            'unhealthy_mask': unhealthy_mask,
            'masked_image': masked_image,
        }
        height = 2
        width = height * len(data)
        plt.figure(figsize=(width, height))
        for i, (title, img) in enumerate(data.items()):
            plt.subplot(1, len(data), i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')

        folder_path = f'./data/IPM_FPM_image/show/'
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(folder_path + f'{id}.png')
        # plt.show()

    return masked_image, msk


def get_IPM_FPM_image(img, id):
    fft_image = get_freq_image(img)
    healthy_mask = np.zeros_like(img)
    unhealthy_mask = np.zeros_like(img)

    json_file_path = f"./data/VerTumor600/cropped_vertebrae/cropped_bbox.json"
    with open(json_file_path, "r") as json_file:
        loaded_data = json.load(json_file)

    bbox = None
    for b in loaded_data:
        if b['id'] == int(id.split('_')[-1]):
            bbox = b['cropped_bbox']
            image_size = b['size']
            if not (image_size[0] == img.shape[0] and image_size[1] == img.shape[1]):
                raise ValueError(f"size != (img.shape[0], img.shape[1])")
            break
    if bbox is None:
        raise ValueError(f"No bounding box data found for id {id}")

    for (label, l, cropped_bbox) in bbox:
        (x_min, y_min, x_max, y_max) = cropped_bbox
        if label == 1:
            unhealthy_mask[y_min:y_max, x_min:x_max] = 1
        else:
            healthy_mask[y_min:y_max, x_min:x_max] = 1

    masked_image, msk = visualize_IPM_FPM_image(img, fft_image, healthy_mask, unhealthy_mask, id)

    return masked_image, msk, unhealthy_mask


def diff_seg_datasets(ROOT_DIR, args):
    if args['arg_num'] == '1':  # diff_training_seg_training.py
        training_dataset = syn_MRIDataset(
            ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/MRI_vertebrae/Train{args["ex_num"]}', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
        testing_dataset = syn_MRIDataset(
            ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/MRI_vertebrae/Test{args["ex_num"]}', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
    elif args['arg_num'] == '2':
        training_dataset = diff_seg_Dataset(
            ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/MRI_vertebrae/Train{args["ex_num"]}', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
        testing_dataset = diff_seg_Dataset(
            ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/MRI_vertebrae/Test{args["ex_num"]}', img_size=args['img_size'],
            random_slice=args['random_slice']
        )
    elif args['arg_num'] == '3':  # Train:spine_trans_cls
        training_dataset = saliency_classifier_Dataset(
            ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/cropped_vertebrae/Train{args["ex_num"]}', img_size=args['img_size'],
        )
        testing_dataset = saliency_classifier_Dataset(
            ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/cropped_vertebrae/Test{args["ex_num"]}', img_size=args['img_size'],
        )
    elif args['arg_num'] == '4':  # Train:spine_trans_cls
        training_dataset = saliency_classifier_Dataset(
            ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/cropped_vertebrae/Train{args["ex_num"]}', img_size=args['img_size'],
        )
        testing_dataset = saliency_classifier_Dataset(
            ROOT_DIR=f'{ROOT_DIR}data/VerTumor600/cropped_vertebrae/Test{args["ex_num"]}', img_size=args['img_size'],
        )
    return training_dataset, testing_dataset


def init_dataset_loader(mri_dataset, args, shuffle=True):
    dataset_loader = torch.utils.data.DataLoader(
        mri_dataset,
        batch_size=args['Batch_Size'],
        shuffle=shuffle,
        # num_workers=4,
        # sampler=sampler,
        # drop_last=True
    )
    return dataset_loader

def spine_dataset_loader(mri_dataset, args, shuffle=True):
    dataset_loader = torch.utils.data.DataLoader(
        mri_dataset,
        batch_size=args['Batch_Size'],
        shuffle=shuffle,
        collate_fn=spine_collate,
        # num_workers=4,
        # sampler=sampler,
        # drop_last=True
    )
    return dataset_loader
class syn_MRIDataset(Dataset):
    """Healthy MRI dataset with synthetic anomaly."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ]
        ) if not transform else transform
        self.mask_transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform
        self.filenames = os.listdir(ROOT_DIR)
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        save_dir = os.path.join(self.ROOT_DIR, self.filenames[idx])
        img_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "image.png")
        mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "mask.png")
        unhealthy_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "unhealthy_mask.png")

        image = cv2.imread(img_name)
        mask = cv2.imread(mask_name)
        unhealthy_mask = cv2.imread(unhealthy_mask_name)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        unhealthy_mask = cv2.cvtColor(unhealthy_mask, cv2.COLOR_BGR2GRAY)

        mask[mask > 0] = 255
        unhealthy_mask[unhealthy_mask > 0] = 255

        input, msk, box_unhealthy_mask = get_IPM_FPM_image(image, self.filenames[idx])
        input = input.astype(np.uint8)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask)
            unhealthy_mask = self.mask_transform(unhealthy_mask)
            input = self.transform(input)
            msk = self.mask_transform(msk)
            box_unhealthy_mask = self.mask_transform(box_unhealthy_mask)

        sample = {'image': image,
                  'input': input,
                  'syn_mask': msk,
                  'box_unhealthy_mask': box_unhealthy_mask,
                  'mask': mask,
                  'unhealthy_mask': unhealthy_mask,
                  "id": self.filenames[idx],
                  "save_dir": save_dir
                  }
        return sample


class diff_seg_Dataset(Dataset):
    """my ver dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_size = img_size
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform
        self.mask_transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform
        self.filenames = os.listdir(ROOT_DIR)
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        save_dir = os.path.join(self.ROOT_DIR, self.filenames[idx])
        img_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "image.png")
        mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "mask.png")
        unhealthy_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "unhealthy_mask.png")

        image = cv2.imread(img_name)
        mask = cv2.imread(mask_name)
        unhealthy_mask = cv2.imread(unhealthy_mask_name)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        unhealthy_mask = cv2.cvtColor(unhealthy_mask, cv2.COLOR_BGR2GRAY)

        _, __, box_unhealthy_mask = get_IPM_FPM_image(image, self.filenames[idx])

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask)
            unhealthy_mask = self.mask_transform(unhealthy_mask)
            box_unhealthy_mask = self.mask_transform(box_unhealthy_mask)

        sample = {'image': image,
                  'mask': mask,
                  'unhealthy_mask': box_unhealthy_mask,
                  "id": self.filenames[idx],
                  "save_dir": save_dir
                  }
        return sample


class common_classifier_Dataset(Dataset):
    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform
        self.mask_transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform
        self.filenames = os.listdir(ROOT_DIR)
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        save_dir = os.path.join(self.ROOT_DIR, self.filenames[int(idx)])
        img_name = os.path.join(self.ROOT_DIR, self.filenames[int(idx)], "image.png")
        # unhealthy_mask_name = os.path.join(self.ROOT_DIR, self.filenames[int(idx)], "mask.png")
        diffusion_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "rec.png")
        out_mask_name = os.path.join(self.ROOT_DIR, self.filenames[idx], "out_mask.png")

        image = cv2.imread(img_name)
        diff = cv2.imread(diffusion_name)
        out_mask = cv2.imread(out_mask_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        out_mask = cv2.cvtColor(out_mask, cv2.COLOR_BGR2GRAY)

        if self.transform:
            image = self.transform(image)
            diff = self.transform(diff)
            out_mask = self.transform(out_mask)

        if "unhealthy" in save_dir:
            label = 1
        else:
            label = 0

        sample = {'image': image,
                  'label': label,
                  'diff': diff,
                  'out_mask': out_mask,
                  "filenames": self.filenames[int(idx)],
                  'save_dir': save_dir
                  }
        return sample

class saliency_classifier_Dataset(Dataset):
    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32)):
        self.img_size = img_size
        self.ROOT_DIR = ROOT_DIR
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Assuming grayscale
        ]) if transform is None else transform

        self.patients = self._load_patients()

    def _load_patients(self):
        # Organize images by patient
        patient_dict = {}
        for filename in os.listdir(self.ROOT_DIR):
            # if filename.endswith('.png'):
            patient_id = filename.split('_')[0]
            if patient_id not in patient_dict:
                patient_dict[patient_id] = []
            patient_dict[patient_id].append(filename)

        # Sort images for each patient by the sequence number
        for patient_id in patient_dict.keys():
            patient_dict[patient_id].sort(key=lambda x: int(x.split('_')[1]))

        return patient_dict

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = list(self.patients.keys())[idx]
        patient_images = self.patients[patient_id]

        images = []
        labels = []
        sals = []
        for filename in patient_images:
            img_path = os.path.join(self.ROOT_DIR, filename, "image.png")
            image = Image.open(img_path).convert('L')
            sal_path = os.path.join(self.ROOT_DIR, filename, f'saliency_map.png')
            sal = Image.open(sal_path).convert('L')
            label = 1 if 'unhealthy' in filename else 0

            if self.transform:
                image = self.transform(image)
                sal = self.transform(sal)

            images.append(image)
            sals.append(sal)
            labels.append(label)

        # images = torch.stack(images)  # Stack images to form a tensor
        # labels = torch.tensor(labels, dtype=torch.long)

        sample = {'images': images,
                  'sals': sals,
                  'labels': labels,
                  'patient_id': patient_id}
        return sample
class common_detection_Dataset(Dataset):

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        self.image_size = img_size
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
             ]
        ) if not transform else transform
        self.mask_transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             transforms.ToTensor(),
             ]
        ) if not transform else transform
        self.filenames = os.listdir(ROOT_DIR)
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        if ".ipynb_checkpoints" in self.filenames:
            self.filenames.remove(".ipynb_checkpoints")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        save_dir = os.path.join(self.ROOT_DIR, self.filenames[int(idx)])
        image = cv2.imread(save_dir)

        if self.transform:
            image = self.transform(image)

        if "unhealthy" in save_dir:
            label = 1
        else:
            label = 0

        sample = {'image': image,
                  'label': label,
                  "filenames": self.filenames[int(idx)],
                  'save_dir': save_dir
                  }
        return sample
