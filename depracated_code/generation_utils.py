import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torch.nn as nn
import os
import time

import argparse

from torchvision import transforms

from typing import  Dict, List, Tuple, Union, Iterable
from tqdm import trange
import torch.nn.functional as F


import matplotlib.pyplot as plt
from torchvision.utils import save_image

from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

from typing import Dict, List, Optional, Union
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    make_list_of_images,
    valid_images,
    validate_preprocess_arguments,
    get_channel_dimension_axis
)

import PIL
from transformers.utils.generic import TensorType
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from transformers.image_utils import ChannelDimension

def to_channel_dimension_format(
    image: torch.Tensor,
    channel_dim: Union[ChannelDimension, str],
    input_channel_dim: Optional[Union[ChannelDimension, str]] = None,
) -> torch.Tensor:
    """
    Converts `image` to the channel dimension format specified by `channel_dim`.

    Args:
        image (`torch.Tensor`):
            The image to have its channel dimension set.
        channel_dim (`ChannelDimension`):
            The channel dimension format to use.
        input_channel_dim (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.

    Returns:
        `torch.Tensor`: The image with the channel dimension set to `channel_dim`.
    """
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"Input image must be of type torch.Tensor, got {type(image)}")

    if input_channel_dim is None:
        input_channel_dim = infer_channel_dimension_format(image)

    target_channel_dim = ChannelDimension(channel_dim)
    if input_channel_dim == target_channel_dim:
        return image

    if target_channel_dim == ChannelDimension.FIRST:
        image = image.permute((2, 0, 1))
    elif target_channel_dim == ChannelDimension.LAST:
        image = image.permute((1, 2, 0))
    else:
        raise ValueError("Unsupported channel dimension format: {}".format(channel_dim))

    return image

def infer_channel_dimension_format(
    image: torch.Tensor, num_channels: Optional[Union[int, Tuple[int, ...]]] = None
) -> ChannelDimension:
    """
    Infers the channel dimension format of `image`.

    Args:
        image (`torch.Tensor`):
            The image to infer the channel dimension of.
        num_channels (`int` or `Tuple[int, ...]`, *optional*, defaults to `(1, 3)`):
            The number of channels of the image.

    Returns:
        The channel dimension of the image.
    """
    num_channels = num_channels if num_channels is not None else (1, 3)
    num_channels = (num_channels,) if isinstance(num_channels, int) else num_channels

    if image.ndim == 3:
        first_dim, last_dim = 0, 2
    elif image.ndim == 4:
        first_dim, last_dim = 1, 3
    else:
        raise ValueError(f"Unsupported number of image dimensions: {image.ndim}")

    if image.shape[first_dim] in num_channels:
        return ChannelDimension.FIRST
    elif image.shape[last_dim] in num_channels:
        return ChannelDimension.LAST

    raise ValueError("Unable to infer channel dimension format")

def preprocess(
    self,
    images: ImageInput,
    do_resize: Optional[bool] = None,
    size: Optional[Dict[str, int]] = None,
    resample: PILImageResampling = None,
    do_rescale: Optional[bool] = None,
    rescale_factor: Optional[float] = None,
    do_normalize: Optional[bool] = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    do_convert_rgb: bool = None,
    data_format: ChannelDimension = ChannelDimension.FIRST,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    **kwargs,
) -> PIL.Image.Image:
    """
    Preprocess an image or batch of images.

    Args:
        images (`ImageInput`):
            Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
            passing in images with pixel values between 0 and 1, set `do_rescale=False`.
        do_resize (`bool`, *optional*, defaults to `self.do_resize`):
            Whether to resize the image.
        size (`Dict[str, int]`, *optional*, defaults to `self.size`):
            Controls the size of the image after `resize`. The shortest edge of the image is resized to
            `size["shortest_edge"]` whilst preserving the aspect ratio. If the longest edge of this resized image
            is > `int(size["shortest_edge"] * (1333 / 800))`, then the image is resized again to make the longest
            edge equal to `int(size["shortest_edge"] * (1333 / 800))`.
        resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`.
        do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
            Whether to rescale the image values between [0 - 1].
        rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
            Rescale factor to rescale the image by if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
            Image mean to normalize the image by if `do_normalize` is set to `True`.
        image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
            Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
        do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
            Whether to convert the image to RGB.
        return_tensors (`str` or `TensorType`, *optional*):
            The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
        data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
            The channel dimension format for the output image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            - Unset: Use the channel dimension format of the input image.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image. If unset, the channel dimension format is inferred
            from the input image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
    """
    do_resize = do_resize if do_resize is not None else self.do_resize
    resample = resample if resample is not None else self.resample
    do_rescale = do_rescale if do_rescale is not None else self.do_rescale
    rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
    do_normalize = do_normalize if do_normalize is not None else self.do_normalize
    image_mean = image_mean if image_mean is not None else self.image_mean
    image_std = image_std if image_std is not None else self.image_std
    do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

    size = size if size is not None else self.size
    size = get_size_dict(size, default_to_square=False)

    images = make_list_of_images(images)

    #validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_processor_keys)

    if not valid_images(images):
        raise ValueError(
            "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
            "torch.Tensor, tf.Tensor or jax.ndarray."
        )

    validate_preprocess_arguments(
        do_rescale=do_rescale,
        rescale_factor=rescale_factor,
        do_normalize=do_normalize,
        image_mean=image_mean,
        image_std=image_std,
        do_resize=do_resize,
        size=size,
        resample=resample,
    )
    # RGBA images are converted to RGB
    if do_convert_rgb:
        images = [convert_to_rgb(image) for image in images]

    #if is_scaled_image(images[0]) and do_rescale:
        # logger.warning_once(
        #     "It looks like you are trying to rescale already rescaled images. If the input"
        #     " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
        # )
        
    if input_data_format is None:
        # We assume that all images have the same channel dimension format.
        input_data_format = infer_channel_dimension_format(images[0])

    transforms=[]
    if do_resize:
        transforms.append(T.Resize((size['height'], size['width']), interpolation=InterpolationMode.BICUBIC))
    if do_rescale:
        transforms.append(T.Lambda(lambda img: img * rescale_factor))
    if do_normalize:
        transforms.append(T.Normalize(mean=image_mean, std=image_std))

    transform = T.Compose(transforms)
    images = [transform(image) for image in images]

    images = [
        to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
    ]

    encoded_outputs = BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)

    return encoded_outputs

def save_adv_image(adv: torch.Tensor, path:str):
    if adv.is_cuda:
        adv = adv.cpu().detach()
    else:
        adv = adv.detach()

    adv = adv.to(torch.float32)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    save_image(adv,path)

from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy

def preprocess_txt(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        This method uses [`BlipImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        if text is not None:
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
        else:
            text_encoding = None
        return text_encoding


def generate(model, max_new_tokens, input, **kwargs):

    if "output_logits" not in kwargs:
        kwargs["output_logits"] = True
    if "return_dict_in_generate" not in kwargs:
        kwargs["return_dict_in_generate"] = True

    kwargs["max_new_tokens"] = max_new_tokens

    return model.generate(**input, **kwargs)

def greedy_generate(model, max_new_tokens, inputs, **kwargs):

    if "output_logits" not in kwargs:
        kwargs["output_logits"] = True
    if "return_dict_in_generate" not in kwargs:
        kwargs["return_dict_in_generate"] = True

    bs = inputs["pixel_values"].shape[0]
    input_ids = (
                torch.LongTensor([[1]])
                .repeat(bs, 1)
                .to("cuda")
            )

    decoder_input_ids = [0]
    logits_hist = []

    for i in range(max_new_tokens):

        if len(decoder_input_ids) > 0:

            decoder_input_ids_torch = torch.tensor(decoder_input_ids).to("cuda").view(bs,-1)

            outs = model(pixel_values=inputs["pixel_values"], input_ids=input_ids, decoder_input_ids=decoder_input_ids_torch)
        else:
            outs = model(pixel_values=inputs["pixel_values"], input_ids=input_ids)
        
        logits = outs.logits
        logits_hist.append(logits[:, -1].detach())
        

        next_token = torch.argmax(logits[:, -1], dim=-1).unsqueeze(-1)
        decoder_input_ids.append(next_token)

    logits_hist = torch.stack(logits_hist, dim=1)

    decoder_input_ids = torch.tensor(decoder_input_ids).to("cuda").view(bs,-1)

    return decoder_input_ids, logits_hist

def single_forward_pass(model, inputs: torch.Tensor, decoder_input_ids: Optional[torch.Tensor], labels: Optional[torch.Tensor], loss_fn):

    bs = inputs["pixel_values"].shape[0]
    device = inputs["pixel_values"].device

    encoder_start_token_id = model.config.text_config.bos_token_id
    decoder_start_token_id = model.config.text_config.decoder_start_token_id

    input_ids = (
                torch.LongTensor([[encoder_start_token_id]])
                .repeat(bs, 1)
                .to(device)
            )


    if decoder_input_ids is None and labels is not None:
        decoder_input_ids = torch.cat([torch.tensor([[decoder_start_token_id]]).to(device).repeat(bs,1), labels], dim=1)
        decoder_input_ids = decoder_input_ids[:,:-1]

    
    if decoder_input_ids is None or len(decoder_input_ids) == decoder_start_token_id:
        decoder_input_ids = torch.tensor([0]*bs).to(device).view(bs,-1)
    elif not (decoder_input_ids[:,0] == decoder_start_token_id).all():
        assert (decoder_input_ids[:,0] == decoder_start_token_id).all(), "Decoder input ids must start with the bos token"

    if labels is None:
        labels = decoder_input_ids[1:]

    out = model(pixel_values=inputs["pixel_values"], input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)

    logits = out.logits
    loss =  loss_fn(logits, labels)
    out_loss = out.loss


    return loss, logits

def test_single_forward_pass():
    pass
    ## Testing


    ###
    # Encoder: Image [32] - [1] (generation_config.encoder_start_token_id)  self.config.text_config.bos_token_id
    # Decoder: [0]              (generation_config.decoder_start_token_id)  self.config.text_config.bos_token_id
    ###

    # inputs = preprocess(self=captioning_processor.image_processor, return_tensors="pt",images=input_, do_rescale=False)
    # inputs.update(inputs_txt)

    # norm_gen = generate(model=captioning_model, max_new_tokens=10, input=inputs)

    # norm_gen_res = captioning_processor.batch_decode(
    #         norm_gen.sequences, skip_special_tokens=True
    #     )

    # greedy_gen_tokens, greedy_gen_logits = greedy_generate(model=captioning_model, max_new_tokens=10, inputs=inputs)

    # greedy_gen_res = captioning_processor.batch_decode(
    #         greedy_gen_tokens, skip_special_tokens=True
    #     )

    # norm_gen_logits = torch.stack(norm_gen.logits, dim=1)

    # greedy_cut_logits = greedy_gen_logits[:,:-1]

    # q = torch.isclose(norm_gen_logits, greedy_cut_logits, atol=7e-2).all()  # It aint low but its something

    # # Make single forward pass to get the logits

    # decoder_input_ids = norm_gen.sequences
    # labels = decoder_input_ids[:,1:]

    # loss, logits = single_forward_pass(captioning_model, inputs, decoder_input_ids=None, labels=labels, loss_fn=loss_fn)

    # q = torch.isclose(norm_gen_logits, logits, atol=7e-2).all()

    # print(norm_gen_res)
    ##


def get_outdir_from_args(args):
    outdir = args.out_dir

    for k in ["loss","eps", "eps_step", "optimizer", "k"]:
        if hasattr(args, k):
            outdir = os.path.join(outdir, f"{k}_{getattr(args, k)}")

    sub_dir_ctr = 0

    joint_path = os.path.join(outdir, f"run_{sub_dir_ctr}")

    while os.path.exists(joint_path):
        sub_dir_ctr += 1
        joint_path = os.path.join(outdir, f"run_{sub_dir_ctr}")

    os.makedirs(joint_path, exist_ok=False)

    with open(os.path.join(joint_path, "args.txt"), "w") as f:
        for k,v in vars(args).items():
            f.write(f"{k}: {v}\n")

    return joint_path