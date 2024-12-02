from typing import List, Tuple, Optional, Callable

import numpy as np
import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import (
    Resize,
    Compose,
    ToTensor,
    Normalize,
    CenterCrop,
    RandomCrop,
    ColorJitter,
    RandomApply,
    GaussianBlur,
    RandomGrayscale,
    RandomResizedCrop,
    RandomHorizontalFlip,
)
from torchvision.transforms.functional import InterpolationMode

from autoaugment import SVHNPolicy, CIFAR10Policy, ImageNetPolicy
from randaugment import RandAugment, RandAugment2, RandAugmentFixMatch

AVAI_CHOICES = [
    "random_flip",
    "random_resized_crop",
    "normalize",
    "instance_norm",
    "random_crop",
    "random_translation",
    "center_crop",  # This has become a default operation during testing
    "cutout",
    "imagenet_policy",
    "cifar10_policy",
    "svhn_policy",
    "randaugment",
    "randaugment_fixmatch",
    "randaugment2",
    "gaussian_noise",
    "colorjitter",
    "randomgrayscale",
    "gaussian_blur",
]

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


class Random2DTranslation:
    """Given an image of (height, width), we resize it to
    (height*1.125, width*1.125), and then perform random cropping.

    Args:
        height (int): target image height.
        width (int): target image width.
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``torchvision.transforms.functional.InterpolationMode.BILINEAR``
    """

    def __init__(self, height, width, p=0.5, interpolation=InterpolationMode.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return F.resize(
                img=img,
                size=[self.height, self.width],
                interpolation=self.interpolation,
            )

        new_width = int(round(self.width * 1.125))
        new_height = int(round(self.height * 1.125))
        resized_img = F.resize(
            img=img, size=[new_height, new_width], interpolation=self.interpolation
        )
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = F.crop(
            img=resized_img, top=y1, left=x1, height=self.height, width=self.width
        )

        return croped_img


class InstanceNormalization:
    """Normalize data using per-channel mean and standard deviation.

    Reference:
        - Ulyanov et al. Instance normalization: The missing in- gredient
          for fast stylization. ArXiv 2016.
        - Shu et al. A DIRT-T Approach to Unsupervised Domain Adaptation.
          ICLR 2018.
    """

    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, img):
        C, H, W = img.shape
        img_re = img.reshape(C, H * W)
        mean = img_re.mean(1).view(C, 1, 1)
        std = img_re.std(1).view(C, 1, 1)
        return (img - mean) / (std + self.eps)


class Cutout:
    """Randomly mask out one or more patches from an image.

    https://github.com/uoguelph-mlrg/Cutout

    Args:
        n_holes (int, optional): number of patches to cut out
            of each image. Default is 1.
        length (int, optinal): length (in pixels) of each square
            patch. Default is 16.
    """

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): tensor image of size (C, H, W).

        Returns:
            Tensor: image with n_holes of dimension
                length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask


class GaussianNoise:
    """Add gaussian noise."""

    def __init__(self, mean=0, std=0.15, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        noise = torch.randn(img.size()) * self.std + self.mean
        return img + noise


def build_transform(
    dataset_name: str,
    transform_choices: List[str] | Tuple[str],
    input_size: Tuple[int],
    pixel_mean: Tuple[float],
    pixel_std: Tuple[float],
    interpolation: str,
    crop_padding: Optional[int | Tuple[int]],
    rrcrop_scale: Tuple[float],
    cutout_n: int,
    cutout_len: int,
    gn_mean: Tuple[float],
    gn_std: Tuple[float],
    randaug_n: int,
    randaug_m: int,
    colorjitter_b: float | Tuple[float, float],
    colorjitter_c: float | Tuple[float, float],
    colorjitter_s: float | Tuple[float, float],
    colorjitter_h: float | Tuple[float, float],
    rgs_p: float,
    gb_k: float,
    gb_p: float,
    is_train: bool = True,
    use_transform: bool = True,
):
    if not use_transform:
        print("Note: no transform is applied!")
        return None

    choices = transform_choices
    for choice in choices:
        assert choice in AVAI_CHOICES

    if dataset_name == "Cifar100":
        target_size = "32x32"
    else:
        target_size = f"{input_size[0]}x{input_size[1]}"

    if dataset_name == "Cifar100":
        normalize = Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    else:
        normalize = Normalize(mean=pixel_mean, std=pixel_std)

    if is_train:
        return _build_transform_train(
            choices=choices,
            input_size=input_size,
            target_size=target_size,
            normalize=normalize,
            interpolation=interpolation,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            crop_padding=crop_padding,
            rrcrop_scale=rrcrop_scale,
            cutout_n=cutout_n,
            cutout_len=cutout_len,
            gn_mean=gn_mean,
            gn_std=gn_std,
            randaug_n=randaug_n,
            randaug_m=randaug_m,
            colorjitter_b=colorjitter_b,
            colorjitter_c=colorjitter_c,
            colorjitter_s=colorjitter_s,
            colorjitter_h=colorjitter_h,
            rgs_p=rgs_p,
            gb_k=gb_k,
            gb_p=gb_p,
        )
    else:
        return _build_transform_test(
            choices=choices,
            input_size=input_size,
            target_size=target_size,
            normalize=normalize,
            interpolation=interpolation,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )


def _build_transform_train(
    choices: List[str] | Tuple[str],
    input_size: Tuple[int],
    target_size: str,
    normalize: Callable,
    interpolation: str,
    pixel_mean: Tuple[float],
    pixel_std: Tuple[float],
    crop_padding: Optional[int | Tuple[int]],
    rrcrop_scale: Tuple[float],
    cutout_n: int,
    cutout_len: int,
    gn_mean: Tuple[float],
    gn_std: Tuple[float],
    randaug_n: int,
    randaug_m: int,
    colorjitter_b: float | Tuple[float, float],
    colorjitter_c: float | Tuple[float, float],
    colorjitter_s: float | Tuple[float, float],
    colorjitter_h: float | Tuple[float, float],
    rgs_p: float,
    gb_k: float,
    gb_p: float,
):
    print("Building transform_train")
    tfm_train = []

    interp_mode = INTERPOLATION_MODES[interpolation]

    # Make sure the image size matches the target size
    conditions = []
    conditions += ["random_crop" not in choices]
    conditions += ["random_resized_crop" not in choices]
    if all(conditions):
        tfm_train += [Resize(input_size, interpolation=interp_mode)]

    if "random_translation" in choices:
        tfm_train += [Random2DTranslation(input_size[0], input_size[1])]

    if "random_crop" in choices:
        tfm_train += [RandomCrop(input_size, padding=crop_padding)]

    if "random_resized_crop" in choices:
        tfm_train += [
            RandomResizedCrop(
                input_size,
                scale=rrcrop_scale,
                interpolation=interp_mode,
            )
        ]

    if "random_flip" in choices:
        tfm_train += [RandomHorizontalFlip()]

    if "imagenet_policy" in choices:
        tfm_train += [ImageNetPolicy()]

    if "cifar10_policy" in choices:
        tfm_train += [CIFAR10Policy()]

    if "svhn_policy" in choices:
        tfm_train += [SVHNPolicy()]

    if "randaugment" in choices:
        n_ = randaug_n
        m_ = randaug_m
        tfm_train += [RandAugment(n_, m_)]

    if "randaugment_fixmatch" in choices:
        n_ = randaug_n
        tfm_train += [RandAugmentFixMatch(n_)]

    if "randaugment2" in choices:
        n_ = randaug_n
        tfm_train += [RandAugment2(n_)]

    if "colorjitter" in choices:
        b_ = colorjitter_b
        c_ = colorjitter_c
        s_ = colorjitter_s
        h_ = colorjitter_h

        tfm_train += [
            ColorJitter(
                brightness=b_,
                contrast=c_,
                saturation=s_,
                hue=h_,
            )
        ]

    if "randomgrayscale" in choices:
        tfm_train += [RandomGrayscale(p=rgs_p)]

    if "gaussian_blur" in choices:
        gb_k, gb_p = gb_k, gb_p
        tfm_train += [RandomApply([GaussianBlur(gb_k)], p=gb_p)]

    tfm_train += [ToTensor()]

    if "cutout" in choices:
        tfm_train += [Cutout(cutout_n, cutout_len)]

    if "normalize" in choices:
        tfm_train += [normalize]

    if "gaussian_noise" in choices:
        tfm_train += [GaussianNoise(gn_mean, gn_std)]

    if "instance_norm" in choices:
        tfm_train += [InstanceNormalization()]

    tfm_train = Compose(tfm_train)

    return tfm_train


def _build_transform_test(
    choices: List[str] | Tuple[str],
    input_size: Tuple[int],
    target_size: str,
    normalize: Callable,
    interpolation: str,
    pixel_mean: Tuple[float],
    pixel_std: Tuple[float],
):
    tfm_test = []

    interp_mode = INTERPOLATION_MODES[interpolation]

    tfm_test += [Resize(max(input_size), interpolation=interp_mode)]

    tfm_test += [CenterCrop(input_size)]

    tfm_test += [ToTensor()]

    if "normalize" in choices:
        tfm_test += [normalize]

    if "instance_norm" in choices:
        tfm_test += [InstanceNormalization()]

    tfm_test = Compose(tfm_test)

    return tfm_test

