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

## Improving generation
What can be done to make images more detailed, realistic and relevant to the topic?

### Specify concept config
Concept config are pharses which are inserted into the system prompt for the LLM used to generate concept lists and prompts for images. Unless you provide it, the default prompt is used, and it might be not specific enough. So adding your concept config to `prompts/concept_config.json` might improve the generation.

Concept config contain the following parameters
|parameter   |meaning                                                                                                     |example           |
|------------|------------------------------------------------------------------------------------------------------------|------------------|
prompt_prefix|used in the prompt for image generation to specify the subject of the image. followed by : in the prompt    |{concept} interior|
concept_name |helps LLM to understand which concepts to suggest                                                           |rooms and offices |
design       |used in system prompt for image prompt generation, includes important aspects that must be described in the prompt| interior design description and style|
example      |example of concepts                                                                                         | bank, customer service area, storage|
shot_length  |shot length of the image to include in prompts                                                              |(long shot or medium shot)|
extra_aspects|some extra aspects to include in prompts. should start with ', '                                            |, camera angle (eye-level), focal length (normal lens or long focus lens, = 24 mm or more), some details about the floor, furniture and objects inside the room|                     

If these explainations do not seem clear, read the system prompts in `prompts/system_prompts` and see where these phrases are inserted to.

### Specify the negative prompt
The default negative prompt might work quite well, especially if you generate interiors, exteriors or items. However, you can also change it. Add the negative prompt for your topic and concept type to `prompts/neg_prompts.json`.
