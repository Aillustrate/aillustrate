import os
import json
import random
import warnings
from typing import List, Union, Tuple
import torch
import diffusers
import transformers
from IPython.display import display


class ImageGenerator:
    def __init__(self, pipe,
                 concept_type = 'interior',
                 negative_prompt = '',
                 topic = '',
                 use_prompt_embeddings = False,
                 num_inference_steps:Union[int, Tuple[int, int]] =(), 
                 guidance_scale:Union[int, Tuple[int, int]] = (),
                 width  = 0, height = 0,
                 root = '',
                 config_path = 'config/image_generation_config.json'
                 ):
        self.pipe = pipe
        self.concept_type = concept_type
        self.config_path =  config_path
        self.set_config(topic, negative_prompt, device,  use_prompt_embeddings, num_inference_steps, guidance_scale, width, height, root)
        self.max_length = self.pipe.tokenizer.model_max_length
    
    def set_config(self, topic, negative_prompt, use_prompt_embeddings, num_inference_steps, guidance_scale, width, height, root):
        if self.config_path:
            with open(self.config_path) as jf:
                config= json.load(jf)
        else:
                config = {}
        self.topic = topic or config.get('topic', topic)
        root = root or config.get('root', root)
        self.root = os.path.join(root, self.topic)
        os.makedirs(self.root, exist_ok=True)
        self.use_prompt_embeddings = config.get('use_prompt_embeddings', use_prompt_embeddings)
        num_inference_steps = num_inference_steps or config.get('num_inference_steps', 25)
        self.num_inference_steps_range = self.set_generation_param_range(num_inference_steps, 'Num inference steps')
        guidance_scale = guidance_scale or config.get('guidance_scale', 7)
        self.guidance_scale_range = self.set_generation_param_range(guidance_scale, 'Guidance scale') 
        self.width  = width or config.get(width, 864)
        self.height = height or config.get(height, 576)
        self.negative_prompt = negative_prompt
        if not self.negative_prompt:
            negative_prompt_path = config.get('negative_prompt_path')
            if negative_prompt_path:
                with open(negative_prompt_path) as jf:
                    self.negative_prompt = json.load(jf)[self.concept_type]
    
        
    def set_generation_param_range(self, generation_param, param_name):
        if type(generation_param) == int:
            return [generation_param]
        if type(generation_param) in [tuple, list]:
            return list(range(*generation_param))
        warnings.warn(f'{param_name} should be either integer or tuple. Got {generation_param}')

    def get_prompt_embeddings(self, prompt, negative_prompt, split_character = ","):
        count_prompt = len(self.pipe.tokenizer.tokenize(prompt))
        count_negative_prompt = len(self.pipe.tokenizer.tokenize(negative_prompt))

        long_kwargs = dict(return_tensors = "pt", truncation = False)
        if count_prompt >= count_negative_prompt:
            input_ids = self.pipe.tokenizer(prompt, **long_kwargs).input_ids.to(self.device)
            shape_max_length = input_ids.shape[-1]
            short_kwargs = dict(return_tensors = "pt", truncation = False, padding = "max_length", max_length = shape_max_length)
            negative_ids = self.pipe.tokenizer(negative_prompt, **short_kwargs).input_ids.to(self.device)
        else:
            negative_ids = self.pipe.tokenizer(negative_prompt, **long_kwargs).input_ids.to(self.device)
            shape_max_length = negative_ids.shape[-1]
            short_kwargs = dict(return_tensors = "pt", truncation = False, padding = "max_length", max_length = shape_max_length)
            input_ids = self.pipe.tokenizer(prompt, **short_kwargs).input_ids.to(self.device)

        concat_embeds = []
        neg_embeds = []
        for i in range(0, shape_max_length, self.max_length):
            concat_embeds.append(self.pipe.text_encoder(input_ids[:, i: i + self.max_length])[0])
            neg_embeds.append(self.pipe.text_encoder(negative_ids[:, i: i + self.max_length])[0])

        return torch.cat(concat_embeds, dim = 1), torch.cat(neg_embeds, dim = 1)

    def save(self, image, name, extension='png'):
        filename = f'{name}.{extension}'
        i = 1
        path = os.path.join(self.root, filename)
        while os.path.exists(path) == True:
            filename = f'{name}{i}.{extension}'
            path = os.path.join(self.root, filename)
            i += 1
        image.save(path)

    def generate(self, prompt, batch_size = 1, start_idx = 0, display_images = True, save = False, series_name = ''):
        seeds = [i for i in range(start_idx , start_idx + batch_size, 1)]
        images = []
        guidance_scale = random.choice(self.guidance_scale_range)
        num_inference_steps = random.choice(self.num_inference_steps_range)
        for count, seed in enumerate(seeds):
            kwargs = dict(width = self.width, height = self.height, 
                      guidance_scale = guidance_scale, num_inference_steps = num_inference_steps,
                      num_images_per_prompt = 1, generator = torch.manual_seed(seed))
            if self.use_prompt_embeddings:
                prompt_embeds, negative_prompt_embeds = self.get_prompt_embeddings(prompt, self.negative_prompt)
                new_img = self.pipe(prompt_embeds = prompt_embeds, negative_prompt_embeds = negative_prompt_embeds, **kwargs).images
            else:
                new_img = self.pipe(prompt = prompt, negative_prompt = self.negative_prompt, **kwargs).images
            images = images + new_img
            for img in new_img:
                if display_images:
                    print(prompt)
                    if len(self.guidance_scale_range) > 1:
                        print(f'Guidance scale: {guidance_scale}')
                    if len(self.num_inference_steps_range) > 1:
                        print(f'Num inference steps: {num_inference_steps}')
                    display(img)
                if save:
                    if not series_name:
                        series_name = ''
                    short_prompt = ' '.join(prompt.split(':')[1].split(' ')[:3])
                    name = f'{series_name}_{short_prompt}_{seed}_gs{guidance_scale}_{num_inference_steps}_steps'
                    self.save(img, name)

        return images

def get_pipe(model_name='', clip_skip=0, text_encoder_name="", device=Nonem torch_dtype=torch.float16, config_path = 'config/image_generation_config.json'):
    if config_path:
        with open(config_path) as jf:
            config= json.load(jf)
    else:
            config = {}
    model_name = model_name or config.get('model_name', 'architectureExterior_v110')
    clip_skip = clip_skip or config.get('clip_skip', 1)
    text_encoder_name = text_encoder_name or config.get('encoder', 'runwayml/stable-diffusion-v1-5')
    device = device or config.get('device', 'cuda')             
                
    if clip_skip > 1:
            text_encoder = transformers.CLIPTextModel.from_pretrained(text_encoder_name, subfolder = "text_encoder", num_hidden_layers = 12 - (clip_skip - 1), torch_dtype = torch_dtype)
            pipe = diffusers.DiffusionPipeline.from_pretrained(model_name, torch_dtype = torch_dtype, text_encoder = text_encoder)
    else:
            pipe = diffusers.DiffusionPipeline.from_pretrained(model_name, torch_dtype = torch_dtype)
    pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    return pipe