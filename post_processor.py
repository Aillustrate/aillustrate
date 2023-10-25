import re
import json
from typing import Dict, List


class PromptProcessor:
    def __init__(self,
                 prompts_path: str,
                 prompts: Dict[str, List[str]] = None,
                 prompt_generator=None,
                 min_words=7,
                 split_prefix=True,
                 intro_words=['sure', 'here', 'certainly', 'ah'],
                 info=True):
        self.prompts_path = prompts_path
        self.prompt_generator = prompt_generator
        self.min_words = min_words
        self.split_prefix = split_prefix
        self.info = info
        self.intro_words = intro_words
        self.special_tokens = [re.compile('<*/?SYS>*'), re.compile('\[/?INST\]')]
        self.prompts = self.load_prompts(prompts, prompts_path)
        self.to_regenerate = {'too short': [], 'has intro': []}

    def load_prompts(self, prompts, prompts_path):
        if prompts:
            return prompts
        with open(prompts_path) as jf:
            return json.load(jf)

    def divide(self, prompt):
        divided_prompt = prompt.split(':')
        prefix = divided_prompt[0]
        if len(divided_prompt) == 0:
            description = prefix
        else:
            description = ':'.join(divided_prompt[1:])
        return prefix, description

    def check_length(self, tokenized_prompt, prompt, concept):
        if len(tokenized_prompt) < self.min_words:
            self.to_regenerate['too short'].append((concept, prompt))
            return False
        return True

    def check_introduction(self, tokenized_prompt, prompt, concept):
        for token in tokenized_prompt:
            if token.lower() in self.intro_words:
                self.to_regenerate['has intro'].append((concept, prompt))
                return False
        return True

    def delete_special_tokens(self, prompt):
        for sp in self.special_tokens:
            return sp.sub('', prompt)

    def add_dot(self, prompt):
        if prompt.endswith('.'):
            return prompt
        elif prompt.endswith(','):
            prompt[-1] = '.'
            return prompt
        return prompt + '.'

    def tokenize(self, prompt):
        return prompt.split(' ')

    def check(self, prompt, concept):
        tokenized_prompt = self.tokenize(prompt)
        return self.check_length(tokenized_prompt, prompt, concept) and self.check_introduction(tokenized_prompt,
                                                                                                prompt, concept)

    def process_prompt(self, prompt, concept):
        processed_prompt = self.delete_special_tokens(prompt)
        processed_prompt = self.add_dot(processed_prompt)
        self.check(processed_prompt, concept)
        return processed_prompt

    def regenerate(self):
        assert self.prompt_generator, 'Prompt generator not provided'
        for problem in self.to_regenerate.values():
            for concept, prompt in problem:
                new_prompt = prompt
                while new_prompt in self.prompts[concept]:
                    new_prompt = \
                    list(self.prompt_generator.generate(save=False, rewrite=True, concepts=[concept]).values())[0][0]
                    new_prompt = self.process_prompt(new_prompt, concept)
                    print(f'{prompt} --> {new_prompt}')
                self.prompts[concept].append(new_prompt)
                self.prompts[concept].remove(prompt)

    def process(self, save=True, auto_regenerate=False):
        processed_prompts = {}
        for concept, promptlist in self.prompts.items():
            processed_prompts[concept] = []
            for prompt in promptlist:
                processed_prompts[concept].append(self.process_prompt(prompt, concept))
        self.prompts = processed_prompts
        if auto_regenerate:
            while any(self.to_regenerate.values()):
                self.regenerate()
        if save:
            with open(self.prompts_path, 'w') as jf:
                json.dump(self.prompts, jf)
        return self.to_regenerate, processed_prompts