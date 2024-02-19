# Automatic image generation pipeline
<table>
   <tr>
<td> <img src='https://github.com/Aillustrate/aillustrate/assets/78635473/9303be7e-a2d1-4d6a-a9d2-ca46a26364ab' width='1000px'> </td>
<td> This tool allows to generate images for various topics and concepts.<br>
By <b>topic</b> we mean the general domain to which images are related <i>(e.g. Innovations and technologies, Home Interior and Lifestyle, Nature and ecology)</i> <br>
By <b>concept</b> we mean what the images depict. We considered 3 types of concepts: <i>interiors, exteriors and items</i>, but this list can be broadened. <br>
See our <a href='ml_system_design_doc/ML_System_Design_Doc_Image_Generation.md'>design doc</a> (RU)  and <a href='https://github.com/Aillustrate/aillustrate/wiki'>wiki</a> (EN) for more details.
</td>
</tr>
</table>

![Image Generation Pipeline](image_generation_pipeline.png)

## How to use

Clone the repository, install requirements and load an image generation model:
```bash
git clone https://github.com/Aillustrate/aillustrate
pip install -r requirements.txt
  ```

Run via command line:
```bash
python main.py --topic=[TOPIC] --concept_type=[CONCEPT_TYPE]
```

Or in Python:
```python
# set your topic and concept type
topic = [your topic] #e.g. Innovations and technologies
concept_type = [concept_type] #e.g. interior

# initialize the pipeline
from aillustrate.pipelines import ParllelPipeline
pipeline = ParllelPipeline(topic, concept_type)

# start the generation process
pipeline.run()
```

Aso see the [notebook](https://github.com/Aillustrate/aillustrate/blob/main/notebooks/example.ipynb) with examples.
