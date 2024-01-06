import argparse
import logging

from diffusers import DiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from image_generator import ImageGenerator, get_pipe
from post_processor import PromptProcessor
from text_generators import ConceptGenerator, PromptGenerator, get_llm
from utils import cleanup, set_logging, update_config

set_logging()


class BasePipeline:
    def __init__(
            self,
            topic,
            concept_type,
            model: AutoModelForCausalLM = None,
            tokenizer: AutoTokenizer = None,
            img_pipe: DiffusionPipeline = None,
            generation_config_concepts_path: str = "config/generation_config_concepts.json",
            generation_config_path: str = "config/generation_config.json",
            image_generation_config_path: str = "config/image_generation_config.json",
    ):
        """
        Pipeline for concept, prompt and image generation
                
        Args:
        param topic: The topic of the images (e.g. 'Innovation and technologies')
        param concept_type: The concept type of the images e.g. 'interior')
        param model: The decode model to generate the concepts and prompts
        param tokenizer: The tokenizer for the model
        param img_pipe: Object of diffusers `DiffusionPipeline` class that will generate the images
        param generation_config_concepts_path: Path to concept generation config
        param generation_config_path: Path to prompt generation config
        param image_generation_config_path: Path to image generation config
        """
        self.topic = topic
        self.concept_type = concept_type
        update_config(self.topic, self.concept_type)
        self.get_models(model, tokenizer, img_pipe)
        self.generation_config_concepts_path = generation_config_concepts_path
        self.generation_config_path = generation_config_path
        self.image_generation_config_path = image_generation_config_path

    def get_models(self, model, tokenizer, img_pipe):
        self.model = model
        self.tokenizer = tokenizer
        self.img_pipe = img_pipe
        cleanup()
        if self.model is None:
            logging.info("Loding LLM")
            self.model, self.tokenizer = get_llm()
        if self.img_pipe is None:
            logging.info("Loading diffusion model")
            self.img_pipe = get_pipe()

    def get_generators(self):
        pass

    def run(
            self,
            min_concepts=150,
            prompt_batch_size=1,
            img_batch_size=1,
    ):
        return None



class SequentialPipeline(BasePipeline):
    def __init__(self, topic, concept_type, **kwargs):
        """Pipeline for concept, prompt and image generation. Prompts and images are generated sequentially (all prompts, then all images)"""
        super().__init__(topic, concept_type, **kwargs)

    def run(
            self,
            min_concepts=150,
            prompt_batch_size=1,
            img_batch_size=1,
    ):
        """
        Generate a set of images by topic and concept type. Prompts and images are generated sequentially

        Args:
        param min_concepts: Total number of concepts to generate (the LLM will go on generating concepts until this amount of concepts is reached)
        param prompt_batch_size: Number of prompts per concept to generate
        param img_batch_size: Number of images per prompt to generate

        """
        concept_generator = ConceptGenerator(
            self.model,
            self.tokenizer,
            min_list_size=min_concepts,
            config_path=self.generation_config_concepts_path,
        )
        prompt_generator = PromptGenerator(
            self.model,
            self.tokenizer,
            batch_size=prompt_batch_size,
            config_path=self.generation_config_path,
        )
        image_generator = ImageGenerator(
            self.img_pipe,
            batch_size=img_batch_size,
            config_path=self.image_generation_config_path,
        )
        logging.info("Generating concepts")
        concept_generator.generate(save=True, rewrite=True)
        logging.info("Generating prompts")
        prompt_generator.generate(save=True, rewrite=True)
        logging.info("Regenerating prompts")
        prompt_processor = PromptProcessor(prompt_generator=prompt_generator)
        prompt_processor.process(save=True, auto_regenerate=True)
        logging.info("Generating images")
        image_generator.generate(save=True)

class ParallelPipeline(BasePipeline):
    def __init__(self, topic, concept_type, **kwargs):
        """Pipeline for concept, prompt and image generation. Prompts and images are generated in parallel (promt + image for each concept)"""
        super().__init__(topic, concept_type, **kwargs)
        
    def run(
            self,
            min_concepts=150,
            prompt_batch_size=1,
            img_batch_size=1,
    ):
        """
        Generate a set of images by topic and concept type. Prompts and images are generated in parallel

        Args:
        param min_concepts: Total number of concepts to generate (the LLM will go on generating concepts until this amount of concepts is reached)
        param prompt_batch_size: Number of prompts per concept to generate
        param img_batch_size: Number of images per prompt to generate

        """
        concept_generator = ConceptGenerator(
            self.model,
            self.tokenizer,
            min_list_size=min_concepts,
            config_path=self.generation_config_concepts_path,
        )
        prompt_generator = PromptGenerator(
            self.model,
            self.tokenizer,
            batch_size=prompt_batch_size,
            config_path=self.generation_config_path,
        )
        image_generator = ImageGenerator(
            self.img_pipe,
            batch_size=img_batch_size,
            config_path=self.image_generation_config_path,
        )
        generated_prompts = prompt_generator.get_collection(rewrite=True, default={})
        image_generator.pipe.set_progress_bar_config(disable=True)
        logging.info("Generating concepts")
        concepts = concept_generator.generate(save=True, rewrite=True)
        logging.info("Generating prompts and images")
        tq = tqdm(concepts)
        for concept in tq:
            tq.set_description(concept)
            prompts = prompt_generator.generate(concepts = [concept], save=True, rewrite=True, show_progress=False)
            generated_prompts.update({concept:prompts})
            prompt_processor = PromptProcessor(prompts=prompts, prompt_generator=prompt_generator)
            _, processed_prompts = prompt_processor.process(save=True, auto_regenerate=True)
            for prompt in processed_prompts:
                image_generator.generate_by_prompt(prompt = prompt, series_name = concept, save=True, display_images=False)
        prompt_generator.save(generated_prompts)
                

if __name__ == '__main__':
    # python main.py --topic='Innovations and technologies' --concept_type='interior' --mode='parallel'
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, required=True)
    parser.add_argument("--concept_type", type=str, required=True)
    parser.add_argument("--mode", type=str, required=False, default='parallel')
    args = parser.parse_args()
    topic, concept_type, mode = args.topic, args.concept_type, args.mode
    if mode == 'sequential':
        full_pipeline = SequentialPipeline(topic, concept_type)
    else:
        full_pipeline = ParallelPipeline(topic, concept_type)
    full_pipeline.run()


