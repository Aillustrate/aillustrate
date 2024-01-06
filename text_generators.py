import gc
import json
import logging
import os
import re
from typing import Any, Dict, List

import torch
from concept_config import ConceptConfig
from pydantic.tools import parse_obj_as
from tqdm.notebook import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import cleanup, parse_concept_config, set_logging

set_logging()


class TextGenerator:
    def __init__(
            self,
            model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            batch_size=1,
            config_path="",
            **kwargs,
    ):
        self.CONCEPT_CONFIG_PATH = "prompts/concept_config.json"
        self.DEFAULT_CONCEPT_CONFIG_PATH = "prompts/default_concept_config.json"
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.config_path = config_path
        self.set_config(**kwargs)
        self.path = os.path.join(self.root,
                                 f"{self.concept_type} ({self.topic}).json")

    def set_config(
            self,
            topic: str = "",
            concept_type: str = "",
            root: str = "",
            device: str = "",
            system_prompt: str = "",
            template: str = "",
            generation_config: Dict[str, Any] = {},
            constraints: Dict[str, Any] = {},
    ):
        """
        Set the configuration for the text generator.

        Args:
        param topic: The topic of the images (e.g. 'Innovation and technologies')
        param concept_type: The concept type of the images e.g. 'interior')
        param root: The root directory where the results of the genration will be stored
        param device: Device to allocate the model on (CPU or CUDA)
        param system_prompt: The system prompt for LLM
        param template: The template for the prompt to LLM
        param generation_config: The text generation config for LLM in a dictionary format
        param constraints: Generation constraints (see `set_constraints` method for more detail)
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
        root = root or self.config.get("root", root)
        self.root = os.path.join(root, self.topic)
        os.makedirs(self.root, exist_ok=True)
        self.device = device or self.config.get("device", "cuda")
        if template:
            self.template = template
        else:
            template_path = self.config["template_path"]
            with open(template_path) as f:
                self.template = f.read()
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            system_prompt_path = self.config["system_prompt_path"]
            with open(system_prompt_path) as f:
                self.system_prompt = f.read()
        self.generation_config = generation_config or self.config.get(
            "generation_config", {}
        )
        self.constraints = constraints or self.config.get("constraints", {})
        self.set_constraints(**constraints)
        concept_config = parse_concept_config(
            self.topic,
            self.concept_type,
            self.CONCEPT_CONFIG_PATH,
            self.DEFAULT_CONCEPT_CONFIG_PATH,
            "concept config",
        )
        self.concept_config = parse_obj_as(ConceptConfig, concept_config)

    def set_constraints(
            self,
            suppress_words: List[str] = None,
            begin_suppress_words: List[str] = None,
            sequence_bias: Dict[str, float] = None,
            forced_eos: str = None,
            bad_words: List[str] = None,
    ):
        """
        Set text generation constraints.
        See https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig for more detail.

        Args:
        param suppress_words: A list of words that will be suppressed at generation
        param begin_suppress_words: A list of words that will be suppressed at the beginning of the generation
        param sequence_bias: Dictionary that maps a sequence of tokens (strings, not numbers) to its bias term
        param forced_eos: The word to force as the last generated token when max_length is reached
        param bad_words: A list of words that are not allowed to be generated
        """
        if suppress_words:
            suppress_tokens = self.get_token_ids(suppress_words)
            self.generation_config.update({"suppress_tokens": suppress_tokens})
        if begin_suppress_words:
            begin_suppress_tokens = self.get_token_ids(begin_suppress_words)
            self.generation_config.update(
                {"begin_suppress_tokens": begin_suppress_tokens}
            )
        if bad_words:
            bad_words_ids = [self.get_token_ids(word) for word in
                             bad_words.split(" ")]
            self.generation_config.update({"bad_words_ids": bad_words_ids})
        if sequence_bias:
            sequence_bias = {
                self.get_tokens_as_tuple(token): value
                for token, value in sequence_bias.items()
            }
            self.generation_config.update({"sequence_bias": sequence_bias})
        if forced_eos:
            forced_eos_token_id = self.get_token_ids(forced_eos)
            self.generation_config.update(
                {"forced_eos_token_id": forced_eos_token_id})

    def set_concept_config(self):
        with open(self.DEFAULT_CONCEPT_CONFIG_PATH) as jf:
            default_concept_config = json.load(jf)
        with open(self.CONCEPT_CONFIG_PATH) as jf:
            all_concept_config = json.load(jf)
            if self.topic in all_concept_config:
                topic_config = all_concept_config[self.topic]
            else:
                logging.warning(
                    f"No concept config provided for {self.topic}. Using default."
                )
                topic_config = default_concept_config
            if self.concept_type in topic_config:
                concept_config = topic_config[self.concept_type]
            elif self.concept_type in default_concept_config:
                logging.warning(
                    f"No concept config provided for {self.concept_type}. Using default config for {self.concept_type}."
                )
                concept_config = topic_config.get("interior")
            else:
                logging.warning(
                    f"No concept config provided for {self.topic} {self.concept_type}. Using default config for interior."
                )
                concept_config = default_concept_config["interior"]
            self.concept_config = parse_obj_as(ConceptConfig, concept_config)

    def make_prompt_to_llm(self, text):
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]
        try:
            return self.tokenizer.apply_chat_template(chat, tokenize=False)
        except:
            return self.template.format(
                system_prompt=self.system_prompt, user_message=text
            )

    def generate_text(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(
            self.device)
        output = self.tokenizer.decode(
            self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.generation_config,
            )[0]
        )
        cleanup()
        return output

    def process_output(self, llm_output):
        pass

    def extract(self, llm_output):
        pass

    def save(self, a):
        with open(self.path, "w") as jf:
            json.dump(a, jf)

    def generate(self, save=True, rewrite=False, sort=True):
        pass

    def get_collection(self, rewrite, default=[]):
        collection = default
        if not rewrite and os.path.exists(self.path) and os.path.getsize(
                self.path) > 0:
            with open(self.path) as jf:
                collection = json.load(jf)
        return collection

    def get_token_ids(self, sequence):
        return self.tokenizer(sequence, add_special_tokens=False).input_ids

    def get_tokens_as_tuple(self, word):
        return tuple(self.get_token_ids([word])[0])


class ConceptGenerator(TextGenerator):
    def __init__(
            self,
            model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            batch_size=20,
            root="concepts",
            min_list_size=0,
            config_path="config/generation_config_concepts.json",
            **kwargs,
    ):
        """
        Args:
        param model: The decoder model to generate the concepts
        param tokenizer: The tokenizer for the model
        param batch_size: Number of concepts to generate in one query to the LLM
        param root: Folder to store the generated concepts
        param min_list_size: Total number of concepts to generate (the LLM will go on generating concepts until this amount of concepts is reached)
        param config_path: Path to concept generator config
        """

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            root=root,
            config_path=config_path,
            **kwargs,
        )
        self.LIST_REGEXP = re.compile(r"\[.+?\]")
        self.LLM_PROMPT_START = "[INST]"
        self.LLM_PROMPT_END = "[/INST]"
        self.SAVE_MESSAGE = f'Concepts saved to "{self.path}"'
        self.min_list_size = min_list_size or self.config.get("min_list_size",
                                                              150)
        if self.min_list_size < self.batch_size:
            self.batch_size = self.min_list_size

    def make_prompt_to_llm(self, text):
        self.system_prompt = self.system_prompt.format(
            n=self.batch_size,
            concept_name=self.concept_config.concept_name,
            topic=self.topic,
            example=self.concept_config.example,
        )
        return self.template.format(prompt=self.system_prompt)

    def process_output(self, llm_output):
        return (
            llm_output.replace(" '", ' ""')
            .replace(",'", ' ,""')
            .replace(",'", ',"')
            .replace("\n", "")
            .replace(",]", "]")
            .replace("/", " ")
            .replace("\\", " ")
            .replace("_", " ")
            .replace("  ", " ")
        )

    def extract(self, llm_output):
        llm_output = self.process_output(llm_output)
        list_string = self.LIST_REGEXP.findall(llm_output)
        if list_string[-1] != self.LLM_PROMPT_END:
            try:
                output_list = json.loads(list_string[-1].lower())
            except:
                output_list = []
        return output_list

    def normalize_list(self, concept_list=[], save=True):
        if concept_list == [] and os.path.exists(self.path):
            with open(self.path) as jf:
                concept_list = json.load(jf)
        for i in range(len(concept_list)):
            concept_list[i] = concept_list[i].lower().strip()
        concept_list = sorted(list(set(concept_list)))
        if save:
            with open(self.path, "w") as jf:
                json.dump(concept_list, jf)
            logging.info(self.SAVE_MESSAGE)
        return concept_list

    def generate_batch(self, save=False, rewrite=False):
        concept_list = self.get_collection(rewrite, default=[])
        prompt = self.make_prompt_to_llm(self.topic)
        llm_output = self.generate_text(prompt)
        new_concepts = self.extract(llm_output)
        concept_list.extend(new_concepts)
        if save:
            self.save(concept_list)
            logging.info(self.SAVE_MESSAGE)
        return concept_list

    def generate(self, save=True, rewrite=True) -> List[str]:
        """
        Generate a list of concepts.

        Args:
        param save: Whether to save the generated concepts
        param rewrite: Whether to retain the previously generated concepts for the current topic and concept type or replace them with the newly generated ones
        return: List of concepts
        """
        concept_list = self.get_collection(rewrite, default=[])
        prev_len = len(concept_list)
        with tqdm(initial=prev_len, total=self.min_list_size) as pbar:
            while len(concept_list) < self.min_list_size:
                try:
                    new_concepts = self.generate_batch(save=False,
                                                       rewrite=True)
                    concept_list.extend(new_concepts)
                    concept_list = self.normalize_list(
                        concept_list=concept_list, save=False
                    )
                finally:
                    pbar.update(len(concept_list) - prev_len)
                    prev_len = len(concept_list)
            if save:
                self.save(concept_list)
                logging.info(self.SAVE_MESSAGE)
            return concept_list


class PromptGenerator(TextGenerator):
    def __init__(
            self,
            model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            batch_size=1,
            root="generated_prompts",
            concepts_root="concepts",
            config_path="config/generation_config.json",
            **kwargs,
    ):
        """
        Args:
        param model: The decode model to generate the prompts
        param tokenizer: The tokenizer for the model
        param batch_size: Number of prompts per concept to generate
        param root: Folder to store the generated prompts
        param concept_root: Folder to take the concepts from
        param config_path: Path to prompt generator config
        """

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            root=root,
            config_path=config_path,
            **kwargs,
        )

        self.RESPONSE_REGEXP = re.compile(r"(?<=\[/INST]).*")
        self.SAVE_MESSAGE = f'Generated prompts saved to "{self.path}"'
        self.concepts_path = os.path.join(
            concepts_root, self.topic,
            f"{self.concept_type} ({self.topic}).json"
        )
        # self.path = os.path.join(self.root, f'{self.concept_type} ({self.topic}).json')

    def make_prompt_to_llm(self, text):
        self.system_prompt = self.system_prompt.format(
            topic=self.topic,
            concept_name=self.concept_config.concept_name,
            design=self.concept_config.design,
            shot_length=self.concept_config.shot_length,
            extra_aspects=self.concept_config.extra_aspects,
        )
        return super().make_prompt_to_llm(text)

    def process(self, llm_output):
        llm_output = re.sub("\n", "", llm_output)
        llm_output = llm_output.replace("</s>", "")
        return llm_output

    def extract(self, llm_output):
        llm_output = self.process(llm_output)
        matches = self.RESPONSE_REGEXP.findall(llm_output)
        if matches:
            extracted_output = matches[0].lstrip()
            return extracted_output[0].lower() + extracted_output[1:]
        return ""

    def generate(
            self,
            save=False,
            rewrite=False,
            display=False,
            sort=True,
            concepts=[],
            rewrite_concept=False,
            show_progress=True
    ) -> Dict[str, str]:
        """
        Generates prompts for images.

        Args:
        param save: Whether to save the generated concepts
        param rewrite: Whether to retain the previously generated prompts for the current topic and concept type or replace them with the newly generated ones
        param display: Whether to output each prompt when it has been generated
        param sort: Whether to sort prompts for each concept
        param concepts: List of concepts to generate the prompts for. If empty, concepts are loaded from the file specified during the initialization
        param rewrite_concept: Whether to retain the previously generated prompts for each concept or replace them with the newly generated ones
        param show_progress: Whether to show progress bar. Recommended to pass `True` for mass generation and `False` for single experiments

        return: Dict of generated prompts for every concept provided
        """
        if not concepts:
            with open(self.concepts_path) as jf:
                concepts = json.load(jf)
        generated_prompts = self.get_collection(rewrite, default={})

        if show_progress:
            tq = tqdm(concepts)
        else:
            tq = concepts
        for concept in tq:
            prompts_for_concept = []
            if show_progress:
                tq.set_description(concept)
            for i in range(self.batch_size):
                prompt_prefix = self.concept_config.prompt_prefix.format(
                    concept=concept
                )
                prompt = self.make_prompt_to_llm(prompt_prefix)
                llm_output = self.generate_text(prompt)
                generated_prompt = f"{prompt_prefix}: {self.extract(llm_output)}"
                prompts_for_concept.append(generated_prompt)
                if display:
                    print(generated_prompt)

            if sort:
                prompts_for_concept = sorted(prompts_for_concept)
            if not rewrite_concept and concept in generated_prompts:
                old_prompts = generated_prompts[concept]
                old_prompts.extend(prompts_for_concept)
                generated_prompts[concept] = list(set(old_prompts))
            else:
                generated_prompts.update({concept: prompts_for_concept})
            if save:
                self.save(generated_prompts)
        if save and show_progress:
            logging.info(self.SAVE_MESSAGE)
        return generated_prompts


def get_llm(
        model_name: str = "",
        device:str = "",
        config_path: str = "config/generation_config.json"
) -> [AutoModelForCausalLM, AutoTokenizer]:
    """
    Get model and tokenizer by name.

    Args:
    param model_name: Huggingface name of the decoder model used for the generation (e.g. 'mistralai/Mistral-7B-Instruct-v0.1')
    param device: Device to allocate the model on (CPU or CUDA)
    param config_path: Path to text generation config
    return: Model and tokenizer

    """
    if config_path:
        with open(config_path) as jf:
            config = json.load(jf)
    else:
        config = {}
    model_name = model_name or config.get(
        "model_name", "mistralai/Mistral-7B-Instruct-v0.1"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    device = device or config.get("device", "cuda")
    model.to(device)
    return model, tokenizer
