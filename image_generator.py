import os
import random
import torch
import diffusers
import transformers
from IPython.display import display


class ImageGenerator:
    def __init__(self, pipe,
                 device ='cuda',
                 negative_prompt = '',
                 use_prompt_embeddings = False,
                 num_inference_steps = 25, guidance_scale = 7,
                 width  = 864, height = 576,
                 dir = ''
                 ):
        self.pipe = pipe
        self.device = device
        self.pipe.to(self.device)
        self.negative_prompt = negative_prompt
        self.max_length = self.pipe.tokenizer.model_max_length
        self.use_prompt_embeddings = use_prompt_embeddings
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)

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
        path = os.path.join(self.dir, filename)
        while os.path.exists(path) == True:
            filename = f'{name}{i}.{extension}'
            path = os.path.join(self.dir, filename)
            i += 1
        image.save(path)

    def generate(self, prompt, batch_size = 1, start_idx = 0, display_images = True, save = True, series_name = ''):
        seeds = [i for i in range(start_idx , start_idx + batch_size, 1)]
        images = []
        guidance_scale = self.guidance_scale
        for count, seed in enumerate(seeds):
            if self.guidance_scale == -1:
                guidance_scale = random.randint(6, 12)
            kwargs = dict(width = self.width, height = self.height, guidance_scale = guidance_scale, num_inference_steps = self.num_inference_steps, num_images_per_prompt = 1, generator = torch.manual_seed(seed))
            if self.use_prompt_embeddings:
                prompt_embeds, negative_prompt_embeds = self.get_prompt_embeddings(prompt, self.negative_prompt)
                new_img = self.pipe(prompt_embeds = prompt_embeds, negative_prompt_embeds = negative_prompt_embeds, **kwargs).images
            else:
                new_img = self.pipe(prompt = prompt, negative_prompt = self.negative_prompt, **kwargs).images
            images = images + new_img
            for img in new_img:
                if display_images:
                    print(prompt)
                    display(img)
                if save:
                    if not series_name:
                        series_name = ''
                    short_prompt = ' '.join(prompt.split(' ')[:3])
                    name = f'{series_name}_{short_prompt}_{seed}_gs{guidance_scale}'
                    self.save(img, name)

        return images

def get_pipe(model_name, clip_skip=1, text_encoder_name="runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16):
        if clip_skip > 1:
            text_encoder = transformers.CLIPTextModel.from_pretrained(text_encoder_name, subfolder = "text_encoder", num_hidden_layers = 12 - (clip_skip - 1), torch_dtype = torch_dtype)
            pipe = diffusers.DiffusionPipeline.from_pretrained(model_name, torch_dtype = torch_dtype, text_encoder = text_encoder)
        else:
            pipe = diffusers.DiffusionPipeline.from_pretrained(model_name, torch_dtype = torch_dtype)
        pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        return pipe