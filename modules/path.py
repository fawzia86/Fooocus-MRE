import os
import json

from os.path import exists


def load_paths():
    path_checkpoints = '../models/checkpoints/'
    path_loras = '../models/loras/'
    path_outputs = '../outputs/'

    if exists('paths.json'):
        with open('paths.json', encoding='utf-8') as paths_file:
            try:
                paths_obj = json.load(paths_file)
                if 'path_checkpoints' in paths_obj:
                    path_checkpoints = paths_obj['path_checkpoints']
                if 'path_loras' in paths_obj:
                    path_loras = paths_obj['path_loras']
                if 'path_outputs' in paths_obj:
                    path_outputs = paths_obj['path_outputs']
            except Exception as e:
                print(e)
                pass
            finally:
                paths_file.close()

    return path_checkpoints, path_loras, path_outputs


path_checkpoints, path_loras, path_outputs = load_paths()

modelfile_path = path_checkpoints if os.path.isabs(path_checkpoints) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_checkpoints))
lorafile_path = path_loras if os.path.isabs(path_loras) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_loras))
temp_outputs_path = path_outputs if os.path.isabs(path_outputs) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_outputs))

os.makedirs(temp_outputs_path, exist_ok=True)

default_base_model_name = 'sd_xl_base_1.0_0.9vae.safetensors'
default_refiner_model_name = 'sd_xl_refiner_1.0_0.9vae.safetensors'
default_lora_name = 'sd_xl_offset_example-lora_1.0.safetensors'
default_lora_weight = 0.5

model_filenames = []
lora_filenames = []


def get_model_filenames(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError("Folder path is not a valid directory.")

    filenames = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            _, file_extension = os.path.splitext(filename)
            if file_extension.lower() in ['.pth', '.ckpt', '.bin', '.safetensors']:
                filenames.append(filename)

    return filenames


def update_all_model_names():
    global model_filenames, lora_filenames
    model_filenames = get_model_filenames(modelfile_path)
    lora_filenames = get_model_filenames(lorafile_path)
    return


update_all_model_names()
