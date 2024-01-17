import gc
import importlib
import json
import logging
import os
import shutil

import pandas as pd
import torch
from tqdm.auto import tqdm


def select_images(path, extension="png"):
    return [fname for fname in os.listdir(path) if fname.endswith(extension)]


def move_files(source, destination, images_only=True):
    if images_only:
        allfiles = select_images(source)
    else:
        allfiles = os.listdir(source)
    for f in allfiles:
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination, f)
        shutil.move(src_path, dst_path)


def cleanup(model=None):
    if model:
        del model
    torch.cuda.empty_cache()
    gc.collect()


def reload(module_path):
    module = importlib.import_module(f'aillustrate.{module_path}')
    importlib.reload(module)
    logging.info(f"{module} reloaded successfully")


def archive(path, result_path="result"):
    """
    Save generated images to zip archive

    param path: Path to source folder with the images
    param result_path: Path to target folder where the archive will be stored
    """
    image_names = select_images(path)
    archive_name = "_".join(path.split("/"))
    folder_to_archive_path = os.path.join(path, archive_name)
    if os.path.exists(folder_to_archive_path):
        folder_to_archive_path = f"{folder_to_archive_path}_1"
    os.makedirs(folder_to_archive_path)
    for fname in tqdm(image_names):
        shutil.move(
            os.path.join(path, fname),
            os.path.join(folder_to_archive_path, fname)
        )
    archive_path = os.path.join(result_path, archive_name)
    shutil.make_archive(archive_path, "zip", folder_to_archive_path)
    logging.info(f"Archive saved to {archive_path}.")


def set_topic(topic, config_dir="config"):
    with open("loras.json") as jf:
        lora_config = json.load(jf).get(topic, {})
    lora_paths = lora_config.get("loras", [])
    trigger_words = lora_config.get("trigger_words", [])
    config_fnames = [
        fname for fname in os.listdir(config_dir) if fname.endswith(".json")
    ]
    for config_fname in config_fnames:
        path = os.path.join(config_dir, config_fname)
        with open(path) as jf:
            config = json.load(jf)
        config.update({"topic": topic})
        if config_fname == "image_generation_config.json":
            config.update({"lora_paths": lora_paths})
            config.update({"trigger_words": trigger_words})
        with open(path, "w") as jf:
            json.dump(config, jf)


def update_config(
        topic,
        concept_type,
        lora_config="config/loras.json",
        generation_config_concepts_path="config/generation_config_concepts.json",
        generation_config_path="config/generation_config.json",
        image_generation_config_path="config/image_generation_config.json",
):
    with open(lora_config) as jf:
        lora_config = json.load(jf).get(topic, {}).get(concept_type, {})
    lora_paths = lora_config.get("lora_paths", [])
    trigger_words = lora_config.get("trigger_words", [])
    config_paths = [
        generation_config_concepts_path,
        generation_config_path,
        image_generation_config_path,
    ]
    for path in config_paths:
        with open(path) as jf:
            config = json.load(jf)
        config.update({"topic": topic, "concept_type": concept_type})
        if path == image_generation_config_path:
            config.update({"lora_paths": lora_paths})
            config.update({"trigger_words": trigger_words})
        with open(path, "w") as jf:
            json.dump(config, jf)


def parse_prompt_config(topic, concept_type, config_path, default_path,
                         config_name):
    with open(default_path) as jf:
        default_prompt_config = json.load(jf)
    with open(config_path) as jf:
        all_prompt_config = json.load(jf)
    if topic in all_prompt_config:
        topic_config = all_prompt_config[topic]
    else:
        topic_config = {}
    if concept_type in topic_config:
        prompt_config = topic_config[concept_type]
    elif concept_type in default_prompt_config:
        logging.warning(
            f"No {config_name} provided for {topic} {concept_type}. Using default {config_name} for {concept_type}."
        )
        prompt_config = default_prompt_config[concept_type]
    else:
        logging.warning(
            f"No {config_name} provided for {topic} {concept_type}. Using general default {config_name}."
        )
        prompt_config = default_prompt_config["default"]
    return prompt_config


def count_images(dir_path="images"):
    counts = []
    for topic in os.listdir(dir_path):
        topic_dir = os.path.join(dir_path, topic)
        for subfolder in os.listdir(topic_dir):
            if not subfolder.startswith("."):
                path = os.path.join(topic_dir, subfolder)
                num_images = len(select_images(path))
                counts.append(
                    {"topic": topic, "sub": subfolder, "images": num_images})
    return pd.DataFrame(counts)


def set_logging():
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        format="[%(asctime)s | %(levelname)s]: %(message)s",
        datefmt="%m.%d.%Y %H:%M:%S",
    )
