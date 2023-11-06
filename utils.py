import gc
import torch
import os
import shutil
import importlib
import json
import random

def move_files(source, destination):
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
    module = importlib.import_module(module_path)
    importlib.reload(module)
    print(f'{module} reloaded successfully')

def archive(path):
    archive_name = '_'.join(path.split('/'))
    shutil.make_archive(archive_name, 'zip', path)

def set_topic(topic, config_dir='config'):
    config_fnames = [fname for fname in os.listdir(config_dir) if fname.endswith('.json')]
    for config_fname in config_fnames:
        path = os.path.join(config_dir, config_fname)
        with open(path) as jf:
            config = json.load(jf)
        config.update({'topic':topic})
        with open(path, 'w') as jf:
            json.dump(config, jf)
    