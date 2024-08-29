import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, GenerationConfig
import torch.nn as nn

from torchvision import transforms

from enum import IntEnum, auto
from typing import Any, Dict, List, Tuple, Union

from fastchat.model import get_conversation_template

import matplotlib
import matplotlib.pyplot as plt
from torchvision.utils import save_image


"""
internvl_chat/internvl/conversation.py

Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you wish to use it.
If you have any changes in mind, please contribute back so the community can benefit collectively and continue to maintain these valuable templates.
"""

"""
/home/ubuntu/.cache/huggingface/modules/transformers_modules/OpenGVLab/Mini-InternVL-Chat-4B-V1-5/5abc8a829e1c848bcb7cc79f22a70e073f68ba87/modeling_internvl_chat.py
/home/ubuntu/miniforge3/envs/internvl/lib/python3.9/site-packages/transformers/generation/utils.py

commented(removed)  #@torch.no_grad()
"""


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def build_transform_pgd(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
 #       T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
 #       T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def dynamic_preprocess_pgd(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = list(image.size()[1:])
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = T.Resize((target_width, target_height))(image)
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        #resized_img.crop(box)
        """
        box – The crop rectangle, as a (left, upper, right, lower)-tuple.
        The right can also be represented as (left+width)
        and lower can be represented as (upper+height).
        """

        #(img: Tensor, top: int, left: int, height: int, width: int)
        #height = abs(upper-lower) = abs(box[1]-box[3])
        #width = abs(right-left) = abs(box[2]-box[0])
        split_img = T.functional.crop(resized_img, box[1],box[0],np.abs(box[1]-box[3]),np.abs(box[2]-box[0])) 
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = T.Resize((image_size, image_size))(image)
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_from_tensor(image_tensor, input_size=448, max_num=6): #the input is an image tensor not a PIL, comnpared to load_image
    #to_pil = transforms.ToPILImage()
    #image = to_pil(image_tensor.squeeze(0))

    transform = build_transform_pgd(input_size=input_size)
    images = dynamic_preprocess_pgd(image_tensor, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def tensor_load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    new_size = (input_size, input_size) 

    # resize the image
    #resized_image = image.resize(new_size) #needs to be done explicitly because resizing is done in the orgiginal load_image
    
    #convert the image to torch.tensor
    image_tensor = transforms.ToTensor()(image)
    return image_tensor

def chat_pgd(my_self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        my_self.img_context_token_id = img_context_token_id

        template = get_conversation_template(my_self.template)
        template.system_message = my_self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * my_self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = my_self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        logits = generation_output.logits
        generation_output = generation_output.sequences
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0] 
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history, logits
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response, logits

path = 'OpenGVLab/Mini-InternVL-Chat-4B-V1-5'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
tokenizer.padding_side = 'left'

generation_config_pgd = dict(
    #output_hidden_states=True,
    #output_scores=True,
    output_logits=True,
    bos_token_id=tokenizer.bos_token_id,
    return_dict_in_generate=True,
    num_beams=1,
    max_new_tokens=3,
    do_sample=False
    #output_attentions=True
)

##INFERENCE
print('Now do inference on the adv attacked image.')
adv_image= tensor_load_image('./examples/adv_attack.png', max_num=6)
pixel_values= load_image_from_tensor(adv_image).to(torch.bfloat16).cuda()
question = '<image>\nWhat color is the animal in the picture? Answer with just a single word of the color name, e.g: red, brown, green or any other color that best describes the color of the animal in the image.'
response, logits = chat_pgd(model,tokenizer, pixel_values, question, generation_config_pgd)
print(f"ADV:\n{response}")

