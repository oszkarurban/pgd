import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, GenerationConfig
import torch.nn as nn

from torchvision import transforms

from enum import IntEnum, auto
from typing import Any, Dict, List, Tuple, Union, Iterable


import matplotlib.pyplot as plt
from torchvision.utils import save_image

from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

from typing import Dict, List, Optional, Union
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
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
    # PIL RGBA images are converted to RGB
    if do_convert_rgb:
        images = [convert_to_rgb(image) for image in images]

    # All transformations expect numpy arrays.
    # images = [to_numpy_array(image) for image in images]

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

def save_adv_image(adv, path):
    if adv.is_cuda:
        adv = adv.cpu().detach()
    else:
        adv = adv.detach()
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


#INCORRECT PARALELL IMPLEMENTATION
from torch.utils.data import DataLoader, TensorDataset
def fgsm_parallel(
    captioning_model,
    captioning_processor,
    x,
    target,
    eps,
    targeted=True,
    clip_min=None,
    clip_max=None,
    batch_size=16
):
    """Parallel process for all FGSM and PGD attacks."""

    # Prepare the input tensor
    input_ = x.clone().detach_().to("cuda")
    input_.requires_grad_()

    # Preprocess the image inputs
    inputs = preprocess(
        self=captioning_processor.image_processor,
        return_tensors="pt",
        images=input_,
        do_rescale=False
    )
    inputs_copy = inputs.copy()

    # Create a DataLoader for batch processing
    dataset = TensorDataset(target)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    accumulated_loss = 0.0

    # Iterate over batches
    for batch in dataloader:
        batch_target = batch[0]

        # Prepare text inputs for the current batch
        batch_texts = [captioning_processor.tokenizer.decode(token_ids=batch_target[i]) for i in range(batch_target.size(0))] #should here be a skip_special_tokens=True?
        
        # Prepare inputs for each text in the batch
        batch_inputs = []
        for text in batch_texts:
            inputs_txt = preprocess_txt(self=captioning_processor, text=text, return_tensors="pt")
            batch_inputs.append(inputs_txt)
        
        # Merge batch inputs with the original image inputs
        for i, inputs_txt in enumerate(batch_inputs):
            if inputs_txt is not None:
                inputs.update(inputs_txt)
        
        inputs = inputs.to(torch.device("cuda"), torch.float16)

        # Caption generation configuration
        generation_config = dict(
            output_logits=True,
            return_dict_in_generate=True,
            max_new_tokens=1
        )
        
        # Generate captions for the batch
        result = captioning_model.generate(
            **inputs, **generation_config
        )
        response = captioning_processor.batch_decode(
            result.sequences, skip_special_tokens=True
        )
        logits = result.logits

        # Stack logits for batch processing
        logits = torch.vstack(logits)

        # Compute loss for the batch
        for i in range(batch_target.size(0)):
            loss = nn.CrossEntropyLoss()(logits, batch_target[i].unsqueeze(0))
            accumulated_loss += loss

    # After processing the entire batch
    generation_config = dict(
        output_logits=True,
        return_dict_in_generate=True,
        max_new_tokens=target.size()[0]
    )
    result = captioning_model.generate(
        **inputs_copy, **generation_config
    )
    response = captioning_processor.batch_decode(
        result.sequences, skip_special_tokens=True
    )
    print(f"after loop fgsm: {response}")

    if response[0]==captioning_processor.tokenizer.decode(token_ids=target,skip_special_tokens=True):
        print("MATCHED")
        out=input_

        if clip_min is not None or clip_max is not None:
            out.clamp_(min=clip_min, max=clip_max)
        return out
        

    # Average loss across the batch
    loss = accumulated_loss / target.size(0)
    captioning_model.zero_grad()
    loss.backward()

    # Perform either targeted or untargeted attack
    if targeted:
        out = input_ - eps * input_.grad  # .sign()
    else:
        out = input_ + eps * input_.grad  # .sign()

    # If desired, clip the output back to the image domain
    if clip_min is not None or clip_max is not None:
        out.clamp_(min=clip_min, max=clip_max)

    return out


def fgsm_(captioning_model, captioning_processor, x, target, eps, iteration,targeted=True, clip_min=None, clip_max=None):
    """Internal process for all FGSM and PGD attacks."""    
    input_ = x.clone().detach_().to("cuda")
    input_.requires_grad_()

    inputs = preprocess(self=captioning_processor.image_processor, return_tensors="pt",images=input_, do_rescale=False)
    inputs_copy = inputs.copy()

    #TEACHER FORCING
    accumulated_loss=0.0
    for i in range(target.size()[0]):    
        text=captioning_processor.tokenizer.decode(token_ids=target[:i])
        inputs_txt= preprocess_txt(self=captioning_processor,text=text,return_tensors="pt")
        if (inputs is not None) and (inputs_txt is not None):
                inputs.update(inputs_txt)
        inputs.to(torch.device("cuda"), torch.float16)


       # labels = captioning_processor.tokenizer("write hello into a textfield", return_tensors="pt").input_ids
        result = captioning_model.forward(pixel_values=inputs.pixel_values,input_ids=inputs.input_ids, labels=labels.cuda())

        # print(f"resonse FORWARD: {captioning_processor.batch_decode(torch.argmax(result.logits, dim=2).tolist(), skip_special_tokens=True)}")
        
        generation_config = dict(
        output_logits=True,
        return_dict_in_generate=True,
        max_new_tokens=1
        )
        result = captioning_model.generate(
            **inputs, **generation_config
        )
        response = captioning_processor.batch_decode(
            result.sequences, skip_special_tokens=True
        )
        logits = result.logits
       
        logits=torch.vstack(logits)    
        loss = nn.CrossEntropyLoss()(logits,target[i].unsqueeze(0)) 
        accumulated_loss += loss
   
    #AFTER TEACHER FORCING
    generation_config = dict(
        output_logits=True,
        return_dict_in_generate=True,
        max_new_tokens=target.size()[0]
        )
    result = captioning_model.generate(
        **inputs_copy, **generation_config
    )
    response = captioning_processor.batch_decode(
        result.sequences, skip_special_tokens=True
    )
    print(f"after TeachF loop fgsm - {iteration}: {response}")

    if response[0]==captioning_processor.tokenizer.decode(token_ids=target,skip_special_tokens=True):
        print("MATCHED")
        #if desired clip the ouput back to the image domain
        out=input_
        if (clip_min is not None) or (clip_max is not None):
            out.clamp_(min=clip_min, max=clip_max)
        return out

    loss = accumulated_loss / target.size()[0]
    captioning_model.zero_grad()
    loss.backward()

    
    #perfrom either targeted or untargeted attack
    if targeted:
        out = input_ - eps * input_.grad#.sign()
    else:
        out = input_ + eps * input_.grad#.sign()
    
    #if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out


def pgd(captioning_model, captioning_processor, x, target, k, eps, eps_step, targeted, clip_min, clip_max):
    print("NEW PGD")
    x_min = (x - eps).cuda()
    x_max = (x + eps).cuda()
    
    # Randomize the starting point x.
    x_adv = x + eps * (2 * torch.rand_like(x) - 1)
    # Clamp back
    if (clip_min is not None) or (clip_max is not None):
        x_adv.clamp_(min=clip_min, max=clip_max)
    
    for i in range(k):
        # FGSM step
        # We don't clamp here (arguments clip_min=None, clip_max=None) 
        # as we want to apply the attack as defined
        x_adv = fgsm_(captioning_model, captioning_processor, x_adv, target, eps_step, i,targeted)
        # x_adv = fgsm_parallel(captioning_model, captioning_processor, x_adv, target,eps_step, targeted)

        # Projection Step
        x_adv = torch.min(x_max, torch.max(x_min, x_adv))
        
    #if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        x_adv.clamp_(min=clip_min, max=clip_max)
    return x_adv

#MODEL & TOKENIZER
captioning_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
captioning_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16
)
captioning_model.to(torch.device("cuda"))

#LOAD IMAGE
image = Image.open("./examples/speaker.png")
image = image.convert("RGB")
image = transforms.ToTensor()(image)

#DEFAULT CAPTIONING
inputs = captioning_processor(
                images=image, return_tensors="pt"
).to(torch.device("cuda"), torch.float16)
generated_ids = captioning_model.generate(
    **inputs, max_new_tokens=32
)
captions = captioning_processor.batch_decode(
    generated_ids, skip_special_tokens=True
)
print(f"DEFAULT DESCRIPTION WITH MAX TOKENS 32: {captions}")

#PGD
target_text=["a black speaker that says hello"]
input_ids = captioning_processor.tokenizer(target_text, return_tensors="pt").input_ids[0].cuda()

adv = pgd(captioning_model,captioning_processor,image, target=input_ids, k=400, eps=0.2, eps_step=0.35, targeted=True, clip_min=0, clip_max=1.0)

#SAVE
path="./visualizeimages/"+"after_pgd_speaker.png"
save_adv_image(adv,path)


