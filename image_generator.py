import json
import logging
import os
import random
import warnings
from typing import List, Tuple, Union

import diffusers
import torch
import transformers
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from IPython.display import display
from tqdm.notebook import tqdm

from utils import cleanup, parse_concept_config, set_logging


class ImageGenerator:
    def __init__(
            self,
            pipe: DiffusionPipeline,
            batch_size: int = 1,
            negative_prompt: str = "",
            config_path: str = "config/image_generation_config.json",
            **kwargs,
    ):
        """
        Args:

        param pipe: Object of diffusers `DiffusionPipeline` class that will generate the images
        param batch_size: Number of images per prompt to generate
        param negative_prompt: Neagtive prompt that lists what shouldn't be in the image.
        If not provided, default prompt for the current topic and concept_type will be used.
        param config_path: Path to image generation config
        """
        set_logging()
        self.DEFAULT_NEGATIVE_PROMPT_PATH = "prompts/default_negative_prompt.json"
        self.pipe = pipe
        self.batch_size = batch_size
        self.config_path = config_path
        self.set_config(**kwargs)
        self.set_negative_prompt(negative_prompt)
        self.max_length = self.pipe.tokenizer.model_max_length

    def set_config(
            self,
            topic: str = "",
            concept_type: str = "",
            use_prompt_embeddings: bool = False,
            num_inference_steps: Union[int, Tuple[int, int]] = (),
            guidance_scale: Union[int, Tuple[int, int]] = (),
            width: int = 0,
            height: int = 0,
            root_dir: str = "",
            prompts_dir: str = "generated_prompts",
            trigger_words: List[str] = [],
    ):
        """
        Set the configuration for the image generator.

        Args:
        param topic: The topic of the images (e.g. 'Innovation and technologies')
        param concept_type: The concept type of the images e.g. 'interior')
        param use_prompt_embeddings: Whether to use prompt embeddings during the genration
        param num_inference_steps: Number of denoising steps made to generate the images.
        The higher it is the more likely is the output to be high quality.
        Can be a fixed number or a range defined in a tuple. If a range is provided, `num_inference_steps` is randomly picked within the provided limits for each generation.
        param guidance_scale: CFG scale from 0 to 20.
        The higher it is the more closely the model follows the prompt.
        Can be a fixed number or a range defined in a tuple. If a range is provided, `guidance_scale` is randomly picked within the provided limits for each generation.
        param width: Width of the resulting images
        param height: Height of the resulting images
        param root_dir: Folder to store the generated images
        param prompts_dir: Folder to take the prompts from
        param trigger_words: Words to add at the end of each prompts to trigger LoRAs
        """
        if self.config_path:
            with open(self.config_path) as jf:
                self.config = json.load(jf)
        else:
            self.config = {}
        self.topic = topic or self.config.get("topic", topic)
        self.concept_type = concept_type or self.config.get(
            "concept_type", concept_type
        )
        root_dir = root_dir or self.config.get("root", root_dir)
        self.root_dir = os.path.join(root_dir, self.topic, self.concept_type)
        os.makedirs(self.root_dir, exist_ok=True)
        prompts_dir = prompts_dir or self.config.get("prompts_dir",
                                                     prompts_dir)
        self.prompts_path = os.path.join(
            prompts_dir, self.topic, f"{self.concept_type} ({self.topic}).json"
        )
        self.use_prompt_embeddings = self.config.get(
            "use_prompt_embeddings", use_prompt_embeddings
        )
        num_inference_steps = num_inference_steps or self.config.get(
            "num_inference_steps", 25
        )
        self.num_inference_steps_range = self.set_generation_param_range(
            num_inference_steps, "Num inference steps"
        )
        guidance_scale = guidance_scale or self.config.get("guidance_scale", 7)
        self.guidance_scale_range = self.set_generation_param_range(
            guidance_scale, "Guidance scale"
        )
        self.width = width or self.config.get("width", 864)
        self.height = height or self.config.get("height", 576)
        self.trigger_words = trigger_words or self.config.get("trigger_words",
                                                              [])

    def set_negative_prompt(self, negative_prompt):
        if negative_prompt:
            self.negative_prompt = negative_prompt
        else:
            negative_prompt_path = self.config.get("negative_prompt_path")
            if not negative_prompt_path:
                logging.warning("No negative prompt provided. Using default.")
                negative_prompt_path = self.DEFAULT_NEGATIVE_PROMPT_PATH
            self.negative_prompt = parse_concept_config(
                self.topic,
                self.concept_type,
                negative_prompt_path,
                self.DEFAULT_NEGATIVE_PROMPT_PATH,
                "negative prompt",
            )

    def set_generation_param_range(self, generation_param, param_name):
        if type(generation_param) is int:
            return [generation_param]
        if type(generation_param) in [tuple, list]:
            return list(range(*generation_param))
        warnings.warn(
            f"{param_name} should be either integer or tuple. Got {generation_param}"
        )

    def get_prompt_embeddings(self, prompt, negative_prompt,
                              split_character=","):
        count_prompt = len(self.pipe.tokenizer.tokenize(prompt))
        count_negative_prompt = len(
            self.pipe.tokenizer.tokenize(negative_prompt))

        long_kwargs = dict(return_tensors="pt", truncation=False)
        if count_prompt >= count_negative_prompt:
            input_ids = self.pipe.tokenizer(prompt,
                                            **long_kwargs).input_ids.to(
                self.device
            )
            shape_max_length = input_ids.shape[-1]
            short_kwargs = dict(
                return_tensors="pt",
                truncation=False,
                padding="max_length",
                max_length=shape_max_length,
            )
            negative_ids = self.pipe.tokenizer(
                negative_prompt, **short_kwargs
            ).input_ids.to(self.device)
        else:
            negative_ids = self.pipe.tokenizer(
                negative_prompt, **long_kwargs
            ).input_ids.to(self.device)
            shape_max_length = negative_ids.shape[-1]
            short_kwargs = dict(
                return_tensors="pt",
                truncation=False,
                padding="max_length",
                max_length=shape_max_length,
            )
            input_ids = self.pipe.tokenizer(prompt,
                                            **short_kwargs).input_ids.to(
                self.device
            )

        concat_embeds = []
        neg_embeds = []
        for i in range(0, shape_max_length, self.max_length):
            concat_embeds.append(
                self.pipe.text_encoder(input_ids[:, i:i + self.max_length])[0]
            )
            neg_embeds.append(
                self.pipe.text_encoder(negative_ids[:, i:i + self.max_length])[
                    0]
            )

        return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)

    def save(self, image, name, extension="png"):
        filename = f"{name}.{extension}"
        i = 1
        path = os.path.join(self.root_dir, filename)
        while os.path.exists(path) == True:
            filename = f"{name}{i}.{extension}"
            path = os.path.join(self.root_dir, filename)
            i += 1
        image.save(path)

    def generate_by_prompt(
            self,
            prompt: str = None,
            start_idx: int = -1,
            display_images: bool = True,
            save: bool = False,
            series_name: str = "",
            add_trigger_words: bool = True,
            **kwargs,
    ):
        """
        Generates one or multiple images for one prompt.

        Args:
        param prompt: A prompt to generate an image for
        start_idx: Random seed to start iteration from.
        If `batch_size` > 1 and it will increase by 1 for every new generation.
        If set to -1, a new random seed will be randomly picked each time.
        The random seed is passed to the diffusion model to generate the initial noisy vector.
        param display_images: Whether to display each generated image.
        Recommended to pass `False` for mass generation and `True` for single experiments.
        param series_name: Prefix put in the image file when saving
        param add_trigger_words: Whether to add trigger words for LoRA to each prompt
        param **kwargs: Kwargs passed to the image generation pipeline.
        See https://huggingface.co/docs/diffusers/v0.8.0/en/api/pipelines/stable_diffusion#diffusers.StableDiffusionPipeline.__call__ for more detail.
        """
        images = []
        if add_trigger_words:
            prompt = f'{prompt} {",".join(self.trigger_words)}'
        if start_idx == -1:
            seeds = [random.randint(1, int(1e6)) for _ in
                     range(self.batch_size)]
        else:
            seeds = [i for i in
                     range(start_idx, start_idx + self.batch_size, 1)]
        for count, seed in enumerate(seeds):
            guidance_scale = random.choice(self.guidance_scale_range)
            num_inference_steps = random.choice(self.num_inference_steps_range)
            generation_kwargs = dict(
                width=self.width,
                height=self.height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                generator=torch.manual_seed(seed),
            )
            generation_kwargs.update(**kwargs)
            if self.use_prompt_embeddings:
                prompt_embeds, negative_prompt_embeds = self.get_prompt_embeddings(
                    prompt, self.negative_prompt
                )
                new_img = self.pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    **generation_kwargs,
                ).images
            else:
                new_img = self.pipe(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    **generation_kwargs,
                ).images
            images = images + new_img
            for img in new_img:
                if display_images:
                    print(prompt)
                    print(f"Random seed: {seed}")
                    if len(self.guidance_scale_range) > 1:
                        print(f"Guidance scale: {guidance_scale}")
                    if len(self.num_inference_steps_range) > 1:
                        print(f"Num inference steps: {num_inference_steps}")
                    display(img)
                if save:
                    if not series_name:
                        series_name = ""
                    short_prompt = prompt.replace("/", "").split(":")
                    if len(short_prompt) > 1:
                        short_prompt = " ".join(short_prompt[1].split(" ")[:6])
                    else:
                        short_prompt = " ".join(short_prompt[0].split(" ")[:6])
                    name = f"{series_name}_{short_prompt}_{seed}_gs{guidance_scale}_{num_inference_steps}_steps"
                    self.save(img, name)
        return images

    def generate(self, prompt: str = None, save: bool = True, **kwargs):
        """
        Main generation method. Generates image(s) for prompt(s)

        Args:
        param prompt: A prompt to generate an image for.
        If not provided prompts are loaded from a file.
        Recommended to pass this argument for single experiments.
        param save: Whether to save the generated images.
        Recommended to pass `True` for mass generation and `False` for single experiments.
        """
        images = []
        if not prompt:
            with open(self.prompts_path) as jf:
                prompts = json.load(jf)
            tq = tqdm(list(prompts.items()))
            self.pipe.set_progress_bar_config(disable=True)
            for concept, prompts_for_concept in tq:
                tq.set_description(concept)
                for prompt in prompts_for_concept:
                    images.append(
                        self.generate_by_prompt(
                            prompt,
                            save=save,
                            display_images=False,
                            series_name=concept,
                            **kwargs,
                        )
                    )
        else:
            images = self.generate_by_prompt(prompt, save=save, **kwargs)

        if save:
            logging.info(f'Saved to "{self.root_dir}"')

        return images

    def generate_series(self, save: bool = True):
        with open(self.prompts_path) as jf:
            prompts = json.load(jf)
        tq = tqdm(list(prompts.items()))
        self.pipe.set_progress_bar_config(disable=True)
        for concept, prompts_for_concept in tq:
            tq.set_description(concept)
            for prompt in prompts_for_concept:
                images = self.generate(
                    prompt, display_images=False, series_name=concept,
                    save=save
                )
        return images


def get_pipe(
        model_name: str = "",
        clip_skip: int = 0,
        text_encoder_name: str = "",
        device=None,
        torch_dtype=torch.float16,
        lora_paths: List[str] = [],
        config_path: str = "config/image_generation_config.json",
) -> DiffusionPipeline:
    """
    Get the image generation pipeline

    Args:
    param model_name: Diffusion model name (e.g. 'architectureExterior_v110')
    param clip_skip: Number of CLIP layers which are skipped during the generation
    param text_encoder_name: Name of the model from which the text encoder is taken (if `clip_skip` > 1)
    param device: Device to allocate the model on (CPU or CUDA)
    param torch_dtype: Dtype of torch tensors (float16 or float32)
    param lora_paths: Paths to LoRA that will be added to the diffuison model
    param config_path: Path to image generation config

    return: The image generation pipeline (Object of `diffusers.DiffusionPipeline` class)


    """
    set_logging()
    if config_path:
        with open(config_path) as jf:
            config = json.load(jf)
    else:
        config = {}
    model_name = model_name or config.get("model_name",
                                          "architectureExterior_v110")
    clip_skip = clip_skip or config.get("clip_skip", 1)
    text_encoder_name = text_encoder_name or config.get(
        "encoder", "runwayml/stable-diffusion-v1-5"
    )
    device = device or config.get("device", "cuda")
    lora_paths = lora_paths or config.get("lora_paths", [])
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    )
    kwargs = {"torch_dtype": torch_dtype}

    if clip_skip > 1:
        text_encoder = transformers.CLIPTextModel.from_pretrained(
            text_encoder_name,
            subfolder="text_encoder",
            num_hidden_layers=12 - (clip_skip - 1),
            torch_dtype=torch_dtype,
        )
        kwargs.update({"text_encoder": "text_encoder"})
    pipe = diffusers.DiffusionPipeline.from_pretrained(model_name, **kwargs)
    pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config
    )
    pipe.to(device)
    for lora_path in lora_paths:
        pipe.load_lora_weights(".", weight_name=lora_path)
        logging.info(f"Added lora {lora_path}.")
    return pipe
