from typing import Dict

import cv2
import albumentations as albu
from albumentations import pytorch as AT
import numpy as np
import torch


tta_lst = [albu.NoOp(always_apply=True),
            albu.HorizontalFlip(always_apply=True),
            albu.VerticalFlip(always_apply=True),
            albu.Compose([albu.HorizontalFlip(always_apply=True),  albu.VerticalFlip(always_apply=True)])]

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Batch of images of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor_copy = tensor.clone().transpose(1, 0)
        for t, m, s in zip(tensor_copy, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor_copy.transpose(1, 0)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, array):
        """
        Args:
            array (np.array): image of size (H, W, C) to be normalized.
        Returns:
            array: Normalized image.
        """
        return (array - self.mean) / self.std


output_format = {
    "none": lambda array: array,
    "int": lambda array: torch.IntTensor(array),
    "float": lambda array: torch.FloatTensor(array),
    "float16": lambda array: torch.HalfTensor(array),
    "long": lambda array: torch.LongTensor(array),
}

rgby_mean = [0.08123, 0.05293, 0.05398, 0.08153]
rgby_std  = [0.13028, 0.08611, 0.14256, 0.12620]

normalization = {
    "none": lambda array: array,
    "default": lambda array: albu.Normalize(
        mean=rgby_mean,
        std=rgby_std
    )(image=array)["image"],
    "custom": lambda array: Normalize(
        mean=[69.93894845, 95.63922359, 106.96826772, 109.37320937],
        std=[565.15118794, 524.84993571, 541.58770239, 616.08134931]
    )(array),
    "coco": lambda array: albu.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
    )(image=array)['image'],
    "binary": lambda array: np.array(array > 0, np.float32)
}

denormalization = {
    "none": lambda array: array,
    "default": lambda array: UnNormalize(
        mean=[69.93894845, 95.63922359, 106.96826772, 109.37320937],
        std=[565.15118794, 524.84993571, 541.58770239, 616.08134931]
    )(array),
    "custom": lambda array: UnNormalize(
        mean=[69.93894845, 95.63922359, 106.96826772, 109.37320937],
        std=[565.15118794, 524.84993571, 541.58770239, 616.08134931]
    )(array)
}

brit_block = albu.OneOf([
    albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1),
    albu.Blur(blur_limit=3, p=0.7),
    albu.MultiplicativeNoise(per_channel=True, p=1),
    # albu.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.5)
], p=0.6)

spatial_block = albu.Compose([
    albu.RandomRotate90(p=0.5),
    albu.Flip(p=0.5),
    albu.Transpose(p=0.5),
    albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=30, border_mode=0, p=0.4),
    albu.ElasticTransform(alpha=512 * 2, sigma=512 * 0.15, alpha_affine=512 * 0.15, border_mode=0, p=0.2),
    # albu.ElasticTransform()

])


                          # alpha=self.after_crop_size * 2,
#                                   sigma=self.after_crop_size * 0.15,
#                                   alpha_affine=self.after_crop_size * 0.15)
# ])
augmentations = {
    "strong": albu.Compose(
        [
            albu.HorizontalFlip(),
            albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=30, p=0.4),
            albu.ElasticTransform(),
            albu.GaussNoise(),
            albu.OneOf(
                [
                    albu.CLAHE(clip_limit=2),
                    albu.IAASharpen(),
                    albu.RandomBrightnessContrast(),
                    albu.RandomGamma(),
                    albu.MedianBlur(),
                ],
                p=0.5,
            ),
            albu.OneOf([albu.RGBShift(), albu.HueSaturationValue()], p=0.5),
        ]
    ),
    "medium": albu.Compose([albu.Flip(),
                            albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=30, p=0.4),
                            ]),
    "new": albu.Compose([
                         albu.RandomRotate90(p=0.5),
                         albu.Flip(p=0.5),
                         albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=30, p=0.4),
                         albu.core.composition.PerChannel(
                                 albu.OneOf([
                                     albu.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.25),
                                     albu.Blur(blur_limit=3, p=.05)
                                 ]), p=1.0),
                         albu.Cutout(max_h_size=70),
                         # albu.ChannelDropout(p=0.05)

                         ]),
    "new_dropmask": albu.Compose([

        albu.RandomResizedCrop(1024, 1024, (0.8, 1)),
        albu.MaskDropout(max_objects=10, p=0.9),
        albu.RandomRotate90(p=0.5),
        albu.Flip(p=0.5),
        albu.Transpose(p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=30, p=0.4),
        albu.core.composition.PerChannel(
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
                # albu.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.25),
                albu.Blur(blur_limit=3, p=.1)
            ]), p=1.0),
        albu.Cutout(max_h_size=70),
        # albu.ChannelDropout(p=0.05)

    ]),
    "w_dropchannel": albu.Compose([
        albu.ChannelDropout(channel_drop_range=(1, 1), p=0.7),
        albu.RandomResizedCrop(1024, 1024, (0.7, 1)),
        albu.MaskDropout(max_objects=14, p=0.9),
        albu.RandomRotate90(p=0.5),
        albu.Flip(p=0.5),
        albu.Transpose(p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.4),
        albu.core.composition.PerChannel(
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
                albu.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.25),
                albu.Blur(blur_limit=3, p=.1)
            ]), p=1.0),
        albu.Cutout(max_h_size=100),
        # albu.ChannelDropout(p=0.05)

    ]),

    "morebrit": albu.Compose([
        spatial_block,
        brit_block,
        brit_block
    ]),
    "justbrit": albu.Compose([
        brit_block,
        brit_block
    ]),

    "weak": albu.Compose([albu.HorizontalFlip()]),
    "none": albu.Compose([]),
}

size_augmentations = {
    "none": lambda size: albu.NoOp(),
    "resize": lambda size: albu.Resize(height=size, width=size, interpolation=cv2.INTER_AREA),
    "center": lambda size: albu.CenterCrop(size, size),
    "crop_or_resize": lambda size: albu.OneOf([
        albu.RandomCrop(size, size),
        albu.Resize(height=size, width=size)
    ], p=1),
    "crop": lambda size: albu.RandomCrop(size, size),
    # "sized_crop": lambda size: albu.OneOf([albu.RandomSizedCrop([1300, 1600], size, size), albu.Resize(height=size, width=size)], p=1)
}

maskdropout = albu.Compose([albu.RandomResizedCrop(800, 800, (0.8, 1)),
                            albu.MaskDropout(max_objects=10, p=0.9)],
                            additional_targets={
                                                # 'image1': 'image',
                                                'cell_semantic': 'image'})


def get_transforms(config: Dict):
    selfsupervision = config['supervision']
    mask_dropout = config['mask_dropout']
    size = config["size"]
    scope = config.get("augmentation_scope", "none")
    size_transform = config.get("size_transform", "none")

    images_normalization = config.get("images_normalization", "default")
    masks_normalization = config.get("masks_normalization", "binary")

    images_output_format_type = config.get("images_output_format_type", "float")
    masks_output_format_type = config.get("masks_output_format_type", "long")
    unnorm = config.get("unnorm", False)
    tta = config.get("tta", False)



    aug = albu.Compose(
        [augmentations[scope],
         size_augmentations[size_transform](size)]
    )
    def process(image, masks):
        def real_ret(image, masks):
            r = aug(image=image, masks=masks) if masks is not None else aug(image=image)
            img = r["image"]
            transformed_image = output_format[images_output_format_type](
                normalization[images_normalization](img)
            )
            if unnorm:
                transformed_unnormed = img
                return transformed_image, transformed_unnormed
            if masks is not None:
                premasks = r["masks"]
                transformed_masks = [output_format[masks_output_format_type](
                    normalization[masks_normalization](premask)) for premask in premasks]

                return transformed_image, transformed_masks
            return transformed_image

        if mask_dropout:

            res = maskdropout(image=image, mask=masks[0], cell_semantic=masks[1]) if len(masks) == 2 else maskdropout(image=image, mask=masks[0])
            cell_mask = res['mask']

            image = res['image']

            masks = [cell_mask, res['cell_semantic']] if len(masks) == 2 else  [cell_mask]

        if selfsupervision:
            ress = spatial_block(image=image, masks=masks)
            img1 = ress['image']
            img2 = np.copy(img1)
            # print(ress['masks'])
            out1 = real_ret(img1, ress['masks'])
            out2 = real_ret(img2, None)

            return (out1[0], out2), out1[1]







        return real_ret(image, masks)




    return process


# train_transform = albu.Compose([
#
#     albu.RandomRotate90(p=0.5),
#     albu.Flip(p=0.5),
#     albu.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.5),
#     albu.core.composition.PerChannel(
#         albu.OneOf([
#             albu.MotionBlur(p=.05),
#             albu.MedianBlur(blur_limit=3, p=.05),
#             albu.Blur(blur_limit=3, p=.05), ])
#         , p=1.0),
#     albu.OneOf([
#         albu.CoarseDropout(max_holes=16, max_height=IMAGE_SIZE // 16, max_width=IMAGE_SIZE // 16,
#                                      fill_value=0, p=0.5),
#         albu.GridDropout(ratio=0.09, p=0.5),
#         albu.Cutout(num_holes=8, max_h_size=IMAGE_SIZE // 16, max_w_size=IMAGE_SIZE // 16, p=0.2),
#     ], p=0.5),
#     albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
#     AT.ToTensorV2(),
# ],
#     additional_targets={
#         'r': 'image',
#         'g': 'image',
#         'b': 'image',
#         'y': 'image',
#     }
# )

# test_transform = albu.Compose([
#     albu.ToFloat(max_value=65535.0),
#     albu.Resize(IMAGE_SIZE, IMAGE_SIZE),
#     AT.ToTensorV2(),
# ],
#     additional_targets={
#         'r': 'image',
#         'g': 'image',
#         'b': 'image',
#         'y': 'image',
#     }
# )

# tta_transform = albu.Compose([
#     albu.ToFloat(max_value=65535.0),
#     albu.RandomRotate90(p=0.5),
#     albu.Transpose(p=0.5),
#     albu.Flip(p=0.5),
#     albu.Resize(IMAGE_SIZE, IMAGE_SIZE),
#     AT.ToTensorV2(),
# ],
#     additional_targets={
#         'r': 'image',
#         'g': 'image',
#         'b': 'image',
#         'y': 'image',
#     }
# )
