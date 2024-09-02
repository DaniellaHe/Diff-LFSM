import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
import torchvision.utils


def scale_to_01(array):
    min_val = np.min(array)
    max_val = np.max(array)

    scaled_array = (array - min_val) / (max_val - min_val)
    return scaled_array

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    img = scale_to_01(img)
    mask = scale_to_01(mask)

    if img.ndim == 2:  # Check if image is single channel (grayscale)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert single channel to 3-channel BGR

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def show_sal_on_image_2(img, heatmap, image_weight=0.5):
    # 确保 img 和 heatmap 的形状一致
    assert img.shape == heatmap.shape, "img and heatmap must have the same shape"

    # 将 heatmap 转换为彩色的热图
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # 将 img 转换为三通道图像
    img = cv2.cvtColor(np.uint8(255 * img), cv2.COLOR_GRAY2BGR)

    # 将 img 和 heatmap 融合
    cam = cv2.addWeighted(img, image_weight, heatmap, 1 - image_weight, 0, dtype=cv2.CV_8U)

    return cam

def gridify_output(img, row_size=-1):
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    return torchvision.utils.make_grid(scale_img(img), nrow=row_size, pad_value=-1).cpu().data.permute(
        0, 2, 1
    ).contiguous().permute(
        2, 1, 0
    )


def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd


def load_classifier_checkpoint(param, use_checkpoint, device):
    if not use_checkpoint:
        return torch.load(f'../model/diff-params-ARGS=296/checkpoint/diff/diff_epoch=5.pt', map_location=device)


def load_parameters(args, device):
    rec_model_path = f"./model/params-ARGS=1_num={args['ex_num']}/checkpoint/diff/params-final.pt"
    # seg_model_path = f"./model/params-ARGS=1_num={args['ex_num']}/checkpoint/seg/params-final.pt"
    seg_model_path = f"./model/params-ARGS=1_num={args['ex_num']}/checkpoint/seg/seg_epoch=25.pt"
    print("load rec_model from:", rec_model_path)
    print("load seg_model from:", seg_model_path)
    return torch.load(rec_model_path, map_location=device), \
        torch.load(seg_model_path, map_location=device)


def load_classifier_parameters(device):
    import sys

    if len(sys.argv[1:]) > 0:
        params = sys.argv[1:]
    else:
        params = os.listdir("./model")

    if ".DS_Store" in params:
        params.remove(".DS_Store")

    if params[0] == "CHECKPOINT":
        use_checkpoint = True
        params = params[1:]
    else:
        use_checkpoint = False

    print(params)
    for param in params:
        if param.isnumeric():
            diff_output = load_classifier_checkpoint(param, use_checkpoint, device)
        else:
            raise ValueError(f"Unsupported input {param}")

        if "args" in diff_output:
            args = diff_output["args"]
        else:
            try:
                with open(f'./configs/args{param[17:]}.json', 'r') as f:
                    args = json.load(f)
                args['arg_num'] = param[17:]
                args = defaultdict_from_json(args)
            except FileNotFoundError:
                raise ValueError(f"args{param[17:]} doesn't exist for {param}")

        if "noise_fn" not in args:
            args["noise_fn"] = "gauss"

        args['arg_num'] = param

        return args, diff_output


def scale_to_01(array):
    min_val = np.min(array)
    max_val = np.max(array)

    scaled_array = (array - min_val) / (max_val - min_val)
    return scaled_array


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    img = scale_to_01(img)
    mask = scale_to_01(mask)

    if img.ndim == 2:  # Check if image is single channel (grayscale)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert single channel to 3-channel BGR

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def main():
    pass


if __name__ == '__main__':
    main()
