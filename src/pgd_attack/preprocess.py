"""Differentiable BLIP-2 image preprocessing.

`BlipImageProcessor.preprocess` converts inputs to numpy and breaks the autograd
graph. PGD requires gradients to flow through preprocessing, so we re-implement
the resize / rescale / normalise pipeline using torchvision transforms that keep
the input as a `torch.Tensor` end-to-end.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    make_list_of_images,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.image_transforms import convert_to_rgb
from transformers.utils.generic import TensorType


def _to_channel_first(image: torch.Tensor, source: ChannelDimension) -> torch.Tensor:
    if source == ChannelDimension.FIRST:
        return image
    if source == ChannelDimension.LAST:
        return image.permute(2, 0, 1) if image.ndim == 3 else image.permute(0, 3, 1, 2)
    raise ValueError(f"Unsupported channel dimension format: {source}")


def _infer_channel_dim(image: torch.Tensor, num_channels: Sequence[int] = (1, 3)) -> ChannelDimension:
    if image.ndim == 3:
        first, last = 0, 2
    elif image.ndim == 4:
        first, last = 1, 3
    else:
        raise ValueError(f"Unsupported number of image dimensions: {image.ndim}")
    if image.shape[first] in num_channels:
        return ChannelDimension.FIRST
    if image.shape[last] in num_channels:
        return ChannelDimension.LAST
    raise ValueError("Unable to infer channel dimension format")


def differentiable_preprocess(
    processor,
    images: ImageInput,
    *,
    do_resize: Optional[bool] = None,
    size: Optional[dict[str, int]] = None,
    resample: Optional[PILImageResampling] = None,
    do_rescale: Optional[bool] = None,
    rescale_factor: Optional[float] = None,
    do_normalize: Optional[bool] = None,
    image_mean: Optional[Union[float, Sequence[float]]] = None,
    image_std: Optional[Union[float, Sequence[float]]] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    do_convert_rgb: Optional[bool] = None,
    data_format: ChannelDimension = ChannelDimension.FIRST,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> BatchFeature:
    """Drop-in replacement for `BlipImageProcessor.preprocess` that preserves gradients.

    Tensor inputs already on a CUDA device are kept where they are; only resize,
    rescale and normalise transforms are applied so gradients propagate through.
    """
    do_resize = do_resize if do_resize is not None else processor.do_resize
    resample = resample if resample is not None else processor.resample
    do_rescale = do_rescale if do_rescale is not None else processor.do_rescale
    rescale_factor = rescale_factor if rescale_factor is not None else processor.rescale_factor
    do_normalize = do_normalize if do_normalize is not None else processor.do_normalize
    image_mean = image_mean if image_mean is not None else processor.image_mean
    image_std = image_std if image_std is not None else processor.image_std
    do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else processor.do_convert_rgb
    size = size if size is not None else processor.size
    size = get_size_dict(size, default_to_square=False)

    images = make_list_of_images(images)
    if not valid_images(images):
        raise ValueError("Invalid image type. Expected PIL.Image, numpy.ndarray or torch.Tensor.")
    validate_preprocess_arguments(
        do_rescale=do_rescale, rescale_factor=rescale_factor,
        do_normalize=do_normalize, image_mean=image_mean, image_std=image_std,
        do_resize=do_resize, size=size, resample=resample,
    )
    if do_convert_rgb:
        images = [convert_to_rgb(image) for image in images]

    if input_data_format is None:
        input_data_format = _infer_channel_dim(images[0])

    transforms: list = []
    if do_resize:
        # bicubic + antialias=True so .uint8 images resize cleanly without overshoot
        # see https://github.com/pytorch/vision/issues/2950#issuecomment-2285840770
        transforms.append(
            T.Resize(
                (size["height"], size["width"]),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        )
    if do_rescale:
        transforms.append(T.Lambda(lambda img: img * rescale_factor))
    if do_normalize:
        transforms.append(T.Normalize(mean=image_mean, std=image_std))

    transform = T.Compose(transforms)
    images = [_to_channel_first(transform(image), input_data_format) for image in images]
    return BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)
