import os
import gc
import re
import json
from abc import ABC
from typing import Dict
from tqdm.notebook import tqdm, trange
import torch


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

class TextGenerator(ABC):
    def __init__(self, 
                 topic, 
                 system_prompt, 
                 template, 
                 model, tokenizer, 
                 root,
                 generation_config={}):
        self.topic = topic
        self.system_prompt = system_prompt
        self.template = template
        self.model = model
        self.tokenizer = tokenizer
        self.root = root
        self.generation_config = generation_config
        os.makedirs(self.root, exist_ok=True)
        self.path = os.path.join(self.root, f'{self.topic}.json')
        
    def make_prompt_to_llm(self, text):
        return self.template.format(system_prompt = self.system_prompt, user_message=text)

    def generate_text(self, text):
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids.to('cuda')
        output = self.tokenizer.decode(self.model.generate(input_ids, **self.generation_config)[0])
        cleanup()
        return output
    
    def process_output(self, llm_output):
        pass
    
    def extract(self, llm_output):
        pass
        
    def save(self, new, rewrite):
        with open(self.path, 'w') as jf:
            json.dump(new, jf)
    
    def generate(self, save = True, rewrite = False):
        pass
    
class LocationListGenerator(TextGenerator):
    def __init__(self, 
                 topic, 
                 system_prompt, template, 
                 model, tokenizer, 
                 root = 'locations',
                 generation_config={}):
        super().__init__(topic, system_prompt, template, model, tokenizer, root, generation_config)
        self.LIST_REGEXP = re.compile('\[.+?\]')
        self.LLM_PROMPT_END = '[/INST]'
        self.MAX_LENGTH=1000
        self.DO_SAMPLE=True

    def process_output(self, llm_output):
        return llm_output.replace(" ''", ' ""').replace(",'", ',"').replace('\n', '').replace(',]',']')

    def extract(self, llm_output):
        llm_output = self.process_output(llm_output)
        list_string = self.LIST_REGEXP.findall(llm_output)
        print(f'{self.topic}\n{list_string[-1]}')
        if list_string[-1] != self.LLM_PROMPT_END:
            return json.loads(list_string[-1])
        return []

    def save(self, location_list, rewrite):
        old_list = []
        if rewrite:
            old_list = []
        elif os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path) as jf:
                old_list = json.loads(jf.read())
        old_list.extend(location_list)
        old_list = sorted(list(set(old_list)))
        with open(self.path, 'w') as jf:
            json.dump(old_list, jf)

    def generate(self, save = True, rewrite = False):
        prompt = self.make_prompt_to_llm(self.topic)
        llm_output =  self.generate_text(prompt)
        location_list = self.extract(llm_output)
        if save:
            self.save(location_list, rewrite)
        return location_list

class PromptGenerator(TextGenerator):
    def __init__(self, 
                 topic, 
                 system_prompt, template, 
                 model, tokenizer, 
                 root = 'generated_prompts', 
                 batch_size=5, 
                 locations=[],
                 generation_config={}, 
                 suppress_words='', begin_suppress_words='', sequence_bias:Dict[str, float]={}, forced_eos = '.',
                view='interior'):
        
        super().__init__(topic, system_prompt, template, model, tokenizer, root, generation_config)
        
        self.RESPONSE_REGEXP = re.compile(f'(?<=\[/INST]).*')
        self.MAX_LENGTH=70
        self.DO_SAMPLE=True
        self.VIEW_TEMPLATES = {"interior":"{location} interior",
                              "exterior":"{location} building exterior",
                              "object":"An object found in {location}"}
        self.set_generation_config(suppress_words, begin_suppress_words, sequence_bias, forced_eos)
        self.locations_path = os.path.join('locations', f'{self.topic}.json')
        self.batch_size = batch_size
        self.view = view
        self.view_template = self.VIEW_TEMPLATES[view]
        if not locations:
            with open(self.locations_path) as jf:
                self.locations = json.load(jf)
        else:
            self.locations = locations
        self.path = os.path.join(self.root, f'{self.topic}_{self.view}.json')
        
    
    def get_token_ids(self, sequence):
        return self.tokenizer(sequence, add_special_tokens=False).input_ids
    
    def get_tokens_as_tuple(self, word):
        return tuple(self.get_token_ids([word])[0])
    
    def set_generation_config(self, suppress_words, begin_suppress_words, sequence_bias, forced_eos):
        
        suppress_tokens = self.get_token_ids(suppress_words)
        begin_suppress_tokens = self.get_token_ids(begin_suppress_words)
        forced_eos_token_id = self.get_token_ids(forced_eos)
        sequence_bias = {self.get_tokens_as_tuple(token): value for token, value in sequence_bias.items()}
        bad_words_ids = [self.get_token_ids(word) for word in begin_suppress_words.split(' ')]
        self.generation_config.update({"suppress_tokens":suppress_tokens, 
                                       "begin_suppress_tokens":begin_suppress_tokens,
                                       "forced_eos_token_id":forced_eos_token_id,
                                       "sequence_bias": sequence_bias,
                                       "bad_words_ids": bad_words_ids})

    def process(self, llm_output):
        llm_output = re.sub('\n', '', llm_output)
        llm_output = llm_output.replace('</s>', '')
        return llm_output
    
    def extract(self, llm_output):
        llm_output = self.process(llm_output)
        matches = self.RESPONSE_REGEXP.findall(llm_output)
        if matches:
            return matches[0].lstrip()
        return ''

    def generate(self, save = True, rewrite = False, display=False):
        
        generated_prompts = {}
        if not rewrite and os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path) as jf:
                generated_prompts = json.load(jf)
                    
        loc_tq = tqdm(self.locations)
        for location in loc_tq:
            prompts_for_location = []
            loc_tq.set_description(location)
            for i in range(self.batch_size):
                location_view = self.view_template.format(location=location)
                prompt = self.make_prompt_to_llm(location_view)
                llm_output =  self.generate_text(prompt)
                generated_prompt = f'{location_view}: {self.extract(llm_output)}'
                prompts_for_location.append(generated_prompt)
                if display:
                    print(generated_prompt)
                
            if not rewrite and location in generated_prompts:
                old_prompts = generated_prompts[location]
                old_prompts.extend(prompts_for_location)
                generated_prompts[location] = list(set(old_prompts))
            else:
                generated_prompts.update({location: sorted(prompts_for_location)})
            if save:
                self.save(generated_prompts, rewrite)
        return generated_prompts
    