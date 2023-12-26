# Automatic image generation pipeline
This tool allows to generate images for various topics and concepts.
By **topic** we mean the general domain to which images are related *(e.g. Innovations and technologies, Home Interior and Lifestyle, Nature and ecology)*  
By **concept** we mean what the images depict. We considered 3 types of concepts: *interiors, exteriors and items*, but this list can be broadened.

## How to use

  ```bash
# install requirements and load the model
pip install -r requirements.txt
load_model.sh 149716 architectureExterior_v110
  ```

   ```python
   # set your topic and concept type
   topic = [your topic] #e.g. Innovations and technologies
   concept_type = [concept_type] #e.g. interior

   # initialize the pipeline
   from main import Pipeline
   pipeline = Pipeline()

   # start the generation process
   pipeline.run(topic, concept_type)
   ```
