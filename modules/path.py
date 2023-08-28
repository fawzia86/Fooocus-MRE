import os
import json

from os.path import exists


def load_paths():
    path_checkpoints = '../models/checkpoints/'
    path_loras = '../models/loras/'
    path_embeddings = '../models/embeddings/'
    path_clip_vision = '../models/clip_vision/'
    path_controlnet = '../models/controlnet/'
    path_outputs = '../outputs/'

    if exists('paths.json'):
        with open('paths.json', encoding='utf-8') as paths_file:
            try:
                paths_obj = json.load(paths_file)
                if 'path_checkpoints' in paths_obj:
                    path_checkpoints = paths_obj['path_checkpoints']
                if 'path_loras' in paths_obj:
                    path_loras = paths_obj['path_loras']
                if 'path_embeddings' in paths_obj:
                    path_embeddings = paths_obj['path_embeddings']
                if 'path_clip_vision' in paths_obj:
                    path_clip_vision = paths_obj['path_clip_vision']
                if 'path_controlnet' in paths_obj:
                    path_controlnet = paths_obj['path_controlnet']
                if 'path_outputs' in paths_obj:
                    path_outputs = paths_obj['path_outputs']

            except Exception as e:
                print(e)
            finally:
                paths_file.close()

    return path_checkpoints, path_loras, path_embeddings, path_clip_vision, path_controlnet, path_outputs


path_checkpoints, path_loras, path_embeddings, path_clip_vision, path_controlnet, path_outputs = load_paths()

modelfile_path = path_checkpoints if os.path.isabs(path_checkpoints) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_checkpoints))
lorafile_path = path_loras if os.path.isabs(path_loras) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_loras))
embeddings_path = path_embeddings if os.path.isabs(path_embeddings) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_embeddings))
clip_vision_path = path_clip_vision if os.path.isabs(path_clip_vision) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_clip_vision))
controlnet_path = path_loras if os.path.isabs(path_controlnet) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_controlnet))
temp_outputs_path = path_outputs if os.path.isabs(path_outputs) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_outputs))

os.makedirs(temp_outputs_path, exist_ok=True)

default_base_model_name = 'sd_xl_base_1.0_0.9vae.safetensors'
default_refiner_model_name = 'sd_xl_refiner_1.0_0.9vae.safetensors'
default_lora_name = 'sd_xl_offset_example-lora_1.0.safetensors'
default_clip_vision_name = 'clip_vision_g.safetensors'
default_controlnet_name = 'control-lora-canny-rank256.safetensors'
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
