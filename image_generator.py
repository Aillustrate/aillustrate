import os
import json
import random
import warnings
from tqdm.notebook import tqdm
from typing import List, Union, Tuple
import torch
import diffusers
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import transformers
from IPython.display import display


class ImageGenerator:
    def __init__(self, pipe,
                 concept_type='interior',
                 negative_prompt='',
                 topic='',
                 use_prompt_embeddings=False,
                 num_inference_steps: Union[int, Tuple[int, int]] = (),
                 guidance_scale: Union[int, Tuple[int, int]] = (),
                 width=0, height=0,
                 root_dir='',
                 prompts_dir = 'generated_prompts',
                 config_path='config/image_generation_config.json',
                 ):
        self.pipe = pipe
        self.concept_type = concept_type
        self.config_path = config_path
        self.set_config(topic, negative_prompt, use_prompt_embeddings, num_inference_steps, guidance_scale,
                        width, height, root_dir, prompts_dir)
        self.max_length = self.pipe.tokenizer.model_max_length

    def set_config(self, topic, negative_prompt, use_prompt_embeddings, num_inference_steps, guidance_scale, width,
                   height, root_dir, prompts_dir):
        if self.config_path:
            with open(self.config_path) as jf:
                config = json.load(jf)
        else:
            config = {}
        self.topic = topic or config.get('topic', topic)
        root_dir = root_dir or config.get('root', root_dir)
        self.root_dir = os.path.join(root_dir, self.topic, self.concept_type)
        os.makedirs(self.root_dir, exist_ok=True)
        prompts_dir = prompts_dir or config.get('prompts_dir', prompts_dir)
        self.prompts_path= os.path.join(prompts_dir, self.topic, f'{self.concept_type} ({self.topic}).json')
        self.use_prompt_embeddings = config.get('use_prompt_embeddings', use_prompt_embeddings)
        num_inference_steps = num_inference_steps or config.get('num_inference_steps', 25)
        self.num_inference_steps_range = self.set_generation_param_range(num_inference_steps, 'Num inference steps')
        guidance_scale = guidance_scale or config.get('guidance_scale', 7)
        self.guidance_scale_range = self.set_generation_param_range(guidance_scale, 'Guidance scale')
        self.width = width or config.get(width, 864)
        self.height = height or config.get(height, 576)
        self.negative_prompt = negative_prompt
        if not self.negative_prompt:
            negative_prompt_path = config.get('negative_prompt_path')
            if negative_prompt_path:
                with open(negative_prompt_path) as jf:
                    self.negative_prompt = json.load(jf)[self.topic][self.concept_type]

    def set_generation_param_range(self, generation_param, param_name):
        if type(generation_param) == int:
            return [generation_param]
        if type(generation_param) in [tuple, list]:
            return list(range(*generation_param))
        warnings.warn(f'{param_name} should be either integer or tuple. Got {generation_param}')

    def get_prompt_embeddings(self, prompt, negative_prompt, split_character=","):
        count_prompt = len(self.pipe.tokenizer.tokenize(prompt))
        count_negative_prompt = len(self.pipe.tokenizer.tokenize(negative_prompt))

        long_kwargs = dict(return_tensors="pt", truncation=False)
        if count_prompt >= count_negative_prompt:
            input_ids = self.pipe.tokenizer(prompt, **long_kwargs).input_ids.to(self.device)
            shape_max_length = input_ids.shape[-1]
            short_kwargs = dict(return_tensors="pt", truncation=False, padding="max_length",
                                max_length=shape_max_length)
            negative_ids = self.pipe.tokenizer(negative_prompt, **short_kwargs).input_ids.to(self.device)
        else:
            negative_ids = self.pipe.tokenizer(negative_prompt, **long_kwargs).input_ids.to(self.device)
            shape_max_length = negative_ids.shape[-1]
            short_kwargs = dict(return_tensors="pt", truncation=False, padding="max_length",
                                max_length=shape_max_length)
            input_ids = self.pipe.tokenizer(prompt, **short_kwargs).input_ids.to(self.device)

        concat_embeds = []
        neg_embeds = []
        for i in range(0, shape_max_length, self.max_length):
            concat_embeds.append(self.pipe.text_encoder(input_ids[:, i: i + self.max_length])[0])
            neg_embeds.append(self.pipe.text_encoder(negative_ids[:, i: i + self.max_length])[0])

        return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)

    def save(self, image, name, extension='png'):
        filename = f'{name}.{extension}'
        i = 1
        path = os.path.join(self.root_dir, filename)
        while os.path.exists(path) == True:
            filename = f'{name}{i}.{extension}'
            path = os.path.join(self.root_dir, filename)
            i += 1
        image.save(path)

    def generate(self, prompt, batch_size=1, start_idx=-1, display_images=True, save=False, series_name='', **kwargs):
        if start_idx==-1:
            start_idx = random.randint(1, int(1e6))
        seeds = [i for i in range(start_idx, start_idx + batch_size, 1)]
        images = []
        for count, seed in enumerate(seeds):
            guidance_scale = random.choice(self.guidance_scale_range)
            num_inference_steps = random.choice(self.num_inference_steps_range)
            generation_kwargs = dict(width=self.width, height=self.height,
                          guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                          num_images_per_prompt=1, generator=torch.manual_seed(seed))
            generation_kwargs.update(**kwargs)
            if self.use_prompt_embeddings:
                prompt_embeds, negative_prompt_embeds = self.get_prompt_embeddings(prompt, self.negative_prompt)
                new_img = self.pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                                    **generation_kwargs).images
            else:
                new_img = self.pipe(prompt=prompt, negative_prompt=self.negative_prompt, **generation_kwargs).images
            images = images + new_img
            for img in new_img:
                if display_images:
                    print(prompt)
                    print(f'Random seed: {seed}')   
                    if len(self.guidance_scale_range) > 1:
                        print(f'Guidance scale: {guidance_scale}')
                    if len(self.num_inference_steps_range) > 1:
                        print(f'Num inference steps: {num_inference_steps}')
                    display(img)
                if save:
                    if not series_name:
                        series_name = ''
                    short_prompt = ' '.join(prompt.replace('/', '').split(':')[1].split(' ')[:6])
                    name = f'{series_name}_{short_prompt}_{seed}_gs{guidance_scale}_{num_inference_steps}_steps'
                    self.save(img, name)

        return images
    
    def generate_series(self, save=True, batch_size=1):
        with open(self.prompts_path) as jf:
            prompts = json.load(jf)
        tq = tqdm(list(prompts.items()))
        self.pipe.set_progress_bar_config(disable=True)
        for concept, prompts_for_concept in tq:
            tq.set_description(concept)
            for prompt in prompts_for_concept:
                images = self.generate(prompt, display_images=False, series_name=concept, save=save, batch_size=batch_size)

def get_pipe(model_name='', clip_skip=0, text_encoder_name="", device=None, torch_dtype = torch.float16, config_path = 'config/image_generation_config.json'):
    if config_path:
        with open(config_path) as jf:
            config = json.load(jf)
    else:
        config = {}
    model_name = model_name or config.get('model_name', 'architectureExterior_v110')
    clip_skip = clip_skip or config.get('clip_skip', 1)
    text_encoder_name = text_encoder_name or config.get('encoder', 'runwayml/stable-diffusion-v1-5')
    device = device or config.get('device', 'cuda')
    safety_checker = StableDiffusionSafetyChecker.from_pretrained('CompVis/stable-diffusion-safety-checker')
    kwargs = {'model_name':model_name, 'torch_dtype':torch_dtype, #'safety_checker':safety_checker
             }

    if clip_skip > 1:
        text_encoder = transformers.CLIPTextModel.from_pretrained(text_encoder_name, subfolder="text_encoder",
                                                                  num_hidden_layers=12 - (clip_skip - 1),
                                                                  torch_dtype=torch_dtype)
        kwargs.update({"text_encoder": "text_encoder"})
    pipe = diffusers.DiffusionPipeline.from_pretrained(model_name, **kwargs)
    pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    return pipe
