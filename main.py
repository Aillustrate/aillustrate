import logging

from diffusers import DiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

from text_generators import get_llm, ConceptGenerator, PromptGenerator
from post_processor import PromptProcessor
from image_generator import get_pipe, ImageGenerator
from utils import update_config, cleanup, set_logging


set_logging()
class Pipeline:
    def __init__(self, model:AutoModelForCausalLM=None, 
                 tokenizer:AutoTokenizer=None, 
                 img_pipe:DiffusionPipeline=None,
                 generation_config_concepts_path:str='config/generation_config_concepts.json',
                 generation_config_path:str='config/generation_config.json',
                 image_generation_config_path:str='config/image_generation_config.json'):
        """

        Args:
        param model: The decode model to generate the concepts and prompts
        param tokenizer: The tokenizer for the model
        param img_pipe: Object of diffusers `DiffusionPipeline` class that will generate the images
        param generation_config_concepts_path: Path to concept generation config
        param generation_config_path: Path to prompt generation config
        param image_generation_config_path: Path to image generation config
        """
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
            logging.info('Loding LLM')
            self.model, self.tokenizer = get_llm()
        if self.img_pipe is None:
            logging.info('Loading diffusion model')
            self.img_pipe = get_pipe()
            
    def run(self, topic, concept_type, min_concepts=150, prompt_batch_size=1, img_batch_size=1):
        """
        Generate a set of images from by topic and concept type

        Args:
        param topic: The topic of the images (e.g. 'Innovation and technologies')
        param concept_type: The concept type of the images e.g. 'interior')
        param min_concepts: Total number of concepts to generate (the LLM will go on generating concepts until this amount of concepts is reached)
        param prompt_batch_size: Number of prompts per concept to generate
        param img_batch_size: Number of images per prompt to generate
        
        """
        update_config(topic, concept_type)
        concept_generator = ConceptGenerator(self.model, self.tokenizer, min_list_size=min_concepts, config_path = self.generation_config_concepts_path)
        prompt_generator = PromptGenerator(self.model, self.tokenizer, 
                                           batch_size=prompt_batch_size,
                                           config_path = self.generation_config_path)
        prompt_processor = PromptProcessor(prompt_generator=prompt_generator)
        image_generator = ImageGenerator(self.img_pipe, 
                                         batch_size=img_batch_size,
                                         config_path = self.image_generation_config_path)
        logging.info('Generating concepts')
        concept_generator.generate(save=True, rewrite=True)
        logging.info('Generating prompts')
        prompt_generator.generate(save=True, rewrite=True)
        logging.info('Regenerating prompts')
        prompt_processor.process(save=True, auto_regenerate=True)
        logging.info('Generating images')
        image_generator.generate(save=True)
        
        
        
        
        

