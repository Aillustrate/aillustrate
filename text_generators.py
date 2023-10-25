import os
import gc
import re
import json
import warnings
from abc import ABC
from typing import Dict
from tqdm.notebook import tqdm, trange
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


class TextGenerator(ABC):
    def __init__(self,
                 concept_type,
                 model, tokenizer,
                 topic='',
                 root='',
                 system_prompt='',
                 template='',
                 generation_config={},
                 constraints={},
                 batch_size=1,
                 config_path=''):
        self.concept_type = concept_type
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.config_path = config_path
        self.set_config(topic, root, system_prompt, template, generation_config, constraints)
        self.path = os.path.join(self.root, f'{self.concept_type} ({self.topic}).json')

    def set_config(self, topic, root, system_prompt, template, generation_config, constraints):
        if self.config_path:
            with open(self.config_path) as jf:
                config = json.load(jf)
        else:
            config = {}
        self.topic = topic or config.get('topic', topic)
        root = root or config.get('root', root)
        self.root = os.path.join(root, self.topic)
        os.makedirs(self.root, exist_ok=True)
        if template:
            self.template = template
        else:
            template_path = config['template_path']
            with open(template_path) as f:
                self.template = f.read()
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            system_prompt_path = config['system_prompt_path']
            with open(system_prompt_path) as f:
                self.system_prompt = f.read()
        self.generation_config = generation_config or config.get('generation_config', {})
        self.constraints = constraints or config.get('constraints', {})
        self.set_constraints(**constraints)
        with open('prompts/concept_config.json') as jf:
            self.concept_config = json.load(jf)

    def set_constraints(self, suppress_words=None, begin_suppress_words=None, sequence_bias=None, forced_eos=None,
                        bad_words=None):
        if suppress_words:
            suppress_tokens = self.get_token_ids(suppress_words)
            self.generation_config.update({"suppress_tokens": suppress_tokens})
        if begin_suppress_words:
            begin_suppress_tokens = self.get_token_ids(begin_suppress_words)
            self.generation_config.update({"begin_suppress_tokens": begin_suppress_tokens})
        if bad_words:
            bad_words_ids = [self.get_token_ids(word) for word in bad_words.split(' ')]
            self.generation_config.update({"bad_words_ids": bad_words_ids})
        if sequence_bias:
            sequence_bias = {self.get_tokens_as_tuple(token): value for token, value in sequence_bias.items()}
            self.generation_config.update({"sequence_bias": sequence_bias})

    def make_prompt_to_llm(self, text):
        chat = [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}]
        try:
            return self.template.format(system_prompt=self.system_prompt, user_message=text)
            return self.tokenizer.apply_chat_template(chat, tokenize=False)
        except:
            return self.template.format(system_prompt=self.system_prompt, user_message=text)

    def generate_text(self, text):
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids.to('cuda')
        output = self.tokenizer.decode(
            self.model.generate(input_ids, pad_token_id=self.tokenizer.eos_token_id, **self.generation_config)[0])
        cleanup()
        return output

    def process_output(self, llm_output):
        pass

    def extract(self, llm_output):
        pass

    def save(self, new, rewrite):
        with open(self.path, 'w') as jf:
            json.dump(new, jf)

    def generate(self, save=True, rewrite=False, sort=True):
        pass


class ConceptGenerator(TextGenerator):
    def __init__(self, concept_type,
                 model, tokenizer,
                 topic='',
                 root='concepts',
                 system_prompt='',
                 template='',
                 generation_config={},
                 constraints={},
                 batch_size=20,
                 config_path='config/generation_config_concepts.json'):
        super().__init__(concept_type, model, tokenizer, topic, root, system_prompt, template, generation_config,
                         constraints, batch_size, config_path)
        self.LIST_REGEXP = re.compile('\[.+?\]')
        self.LLM_PROMPT_END = '[/INST]'
        self.MAX_LENGTH = 1000
        self.DO_SAMPLE = True

    def make_prompt_to_llm(self, text):
        self.system_prompt = self.system_prompt.format(
            n=self.batch_size,
            concept=self.concept_config[self.topic][self.concept_type]['concept name'],
            example=self.concept_config[self.topic][self.concept_type]['example'])
        return super().make_prompt_to_llm(text)

    def process_output(self, llm_output):
        return llm_output.replace(" '", ' ""').replace(",'", ' ,""').replace(",'", ',"').replace('\n', '').replace(',]',
                                                                                                                   ']')

    def extract(self, llm_output):
        llm_output = self.process_output(llm_output)
        list_string = self.LIST_REGEXP.findall(llm_output)
        print(f'{self.topic}\n{list_string[-1]}')
        if list_string[-1] != self.LLM_PROMPT_END:
            return json.loads(list_string[-1].lower())
        return []

    def save(self, concept_list, rewrite, sort):
        old_list = []
        if rewrite:
            old_list = []
        elif os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path) as jf:
                old_list = json.loads(jf.read())
        old_list.extend(concept_list)
        old_list = list(set(old_list))
        if sort:
            old_list = sorted(old_list)
        with open(self.path, 'w') as jf:
            json.dump(old_list, jf)

    def generate(self, save=False, rewrite=False, sort=True):
        prompt = self.make_prompt_to_llm(self.topic)
        llm_output = self.generate_text(prompt)
        concept_list = self.extract(llm_output)
        if save:
            self.save(concept_list, rewrite, sort)
        return concept_list


class PromptGenerator(TextGenerator):
    def __init__(self, concept_type,
                 model, tokenizer,
                 topic='',
                 root='generated_prompts',
                 system_prompt='',
                 template='',
                 generation_config={},
                 constraints={},
                 batch_size=1,
                 config_path='config/generation_config.json'):

        super().__init__(concept_type, model, tokenizer, topic, root, system_prompt, template, generation_config,
                         constraints, batch_size, config_path)

        self.RESPONSE_REGEXP = re.compile(f'(?<=\[/INST]).*')
        self.concepts_path = os.path.join('concepts', self.topic, f'{self.concept_type} ({self.topic}).json')
        self.prompt_prefix = self.concept_config[self.topic][self.concept_type]['prompt prefix']
        self.path = os.path.join(self.root, f'{self.concept_type} ({self.topic}).json')

    def get_token_ids(self, sequence):
        return self.tokenizer(sequence, add_special_tokens=False).input_ids

    def get_tokens_as_tuple(self, word):
        return tuple(self.get_token_ids([word])[0])

        if forced_eos:
            forced_eos_token_id = self.get_token_ids(forced_eos)
            self.generation_config.update({"forced_eos_token_id": forced_eos_token_id})

    def make_prompt_to_llm(self, text):
        self.system_prompt = self.system_prompt.format(
            topic=self.topic,
            concept=self.concept_config[self.topic][self.concept_type]['concept name'],
            design=self.concept_config[self.topic][self.concept_type]['design'],
            shot_length=self.concept_config[self.topic][self.concept_type]['shot length'],
            extra_aspects=self.concept_config[self.topic][self.concept_type]['extra aspects'])
        return super().make_prompt_to_llm(text)

    def process(self, llm_output):
        llm_output = re.sub('\n', '', llm_output)
        llm_output = llm_output.replace('</s>', '')
        return llm_output

    def extract(self, llm_output):
        llm_output = self.process(llm_output)
        matches = self.RESPONSE_REGEXP.findall(llm_output)
        if matches:
            extracted_output = matches[0].lstrip()
            return extracted_output[0].lower() + extracted_output[1:]
        return ''

    def generate(self, save=False, rewrite=False, display=False, sort=True, concepts=[], rewrite_concept=False):
        if not concepts:
            with open(self.concepts_path) as jf:
                concepts = json.load(jf)
        generated_prompts = {}
        if not rewrite and os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path) as jf:
                generated_prompts = json.load(jf)

        tq = tqdm(concepts)
        for concept in tq:
            prompts_for_concept = []
            tq.set_description(concept)
            for i in range(self.batch_size):
                prompt_prefix = self.prompt_prefix.format(concept=concept)
                prompt = self.make_prompt_to_llm(prompt_prefix)
                llm_output = self.generate_text(prompt)
                generated_prompt = f'{prompt_prefix}: {self.extract(llm_output)}'
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
                self.save(generated_prompts, rewrite)
        return generated_prompts


def get_llm(model_name='', config_path='generation_config.json'):
    if config_path:
        with open(config_path) as jf:
            config = json.load(jf)
    else:
        config = {}
    model_name = model_name or config.get('model_name', 'mistralai/Mistral-7B-Instruct-v0.1')
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model.to('cuda')
    return model, tokenizer
