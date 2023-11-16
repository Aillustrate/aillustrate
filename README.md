# wonderslide-interior-generation
This tool allows to generate images for various topics and concepts.
By **topic** we mean the general domain to which images are related *(e.g. Innovations and technologies, Home Interior and Lifestyle, Nature and ecology)*  
By **concept** we mean what the images depict. We considered 3 types of concepts: *interiors, exteriors and items*, but this list can be broadened.

## How to use

1. Set your topic and concept type:
   ```python
   from utils import update_config
   topic = [your topic] #e.g. Innovations and technologies
   concept_type = [concept_type] #e.g. interior
   update_config(topic, concept_type)
   ```

2. Load the models:
   ```python
   from text_generators import get_llm
   from image_generator import get_pipe
   model, tokenizer = get_llm()
   pipe = get_pipe()
   ```
   
3. Generate a list of concepts for your topic and concept_type:
   ```python
   from text_generators import ConceptGenerator
   cg = ConceptGenerator(model, tokenizer)
   cg.generate_complete_list();
   ```
   The list of concepts will be stored in `concepts\<concept_type>/<topic>.json`. You can edit it if you want.
   
4. Generate prompts for each of your concepts:
    ```python
    from text_generators import PromptGenerator
    pg = PromptGenerator(model, tokenizer)
    prompts = pg.generate();
    ```
   The list of concepts will be stored in `generated_prompts/<topic>/<oncept type> (topic).json`. You can also edit them.
  
5. Generate images for your prompts:
   ```python
      from image_generator import ImageGenerator
      ig = ImageGenerator(pipe)
      ig.generate_series();
      ```
   You will find the images in `images/<topic>`.
