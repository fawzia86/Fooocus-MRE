import os
import json

from os.path import exists


def load_paths():
    path_checkpoints = '../models/checkpoints/'
    path_loras = '../models/loras/'
    path_embeddings = '../models/embeddings/'
    path_clip_vision = '../models/clip_vision/'
    path_controlnet = '../models/controlnet/'
    path_vae_approx = '../models/vae_approx/'
    path_fooocus_expansion = '../models/prompt_expansion/fooocus_expansion/'
    path_styles = '../sdxl_styles/'
    path_wildcards = '../wildcards/'
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
                if 'path_vae_approx' in paths_obj:
                    path_vae_approx = paths_obj['path_vae_approx']
                if 'path_fooocus_expansion' in paths_obj:
                    path_fooocus_expansion = paths_obj['path_fooocus_expansion']
                if 'path_styles' in paths_obj:
                    path_styles = paths_obj['path_styles']
                if 'path_wildcards' in paths_obj:
                    path_wildcards = paths_obj['path_wildcards']
                if 'path_outputs' in paths_obj:
                    path_outputs = paths_obj['path_outputs']

            except Exception as e:
                print('load_paths, e: ' + str(e))
            finally:
                paths_file.close()

    return path_checkpoints, path_loras, path_embeddings, path_clip_vision, path_controlnet, path_vae_approx, path_fooocus_expansion, path_styles, path_wildcards, path_outputs


path_checkpoints, path_loras, path_embeddings, path_clip_vision, path_controlnet, path_vae_approx, path_fooocus_expansion, path_styles, path_wildcards, path_outputs = load_paths()

modelfile_path = path_checkpoints if os.path.isabs(path_checkpoints) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_checkpoints))
lorafile_path = path_loras if os.path.isabs(path_loras) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_loras))
embeddings_path = path_embeddings if os.path.isabs(path_embeddings) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_embeddings))
clip_vision_path = path_clip_vision if os.path.isabs(path_clip_vision) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_clip_vision))
controlnet_path = path_controlnet if os.path.isabs(path_controlnet) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_controlnet))
vae_approx_path = path_vae_approx if os.path.isabs(path_vae_approx) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_vae_approx))
fooocus_expansion_path = path_fooocus_expansion if os.path.isabs(path_fooocus_expansion) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_fooocus_expansion))
styles_path = path_styles if os.path.isabs(path_styles) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_styles))
wildcards_path = path_wildcards if os.path.isabs(path_wildcards) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_wildcards))
temp_outputs_path = path_outputs if os.path.isabs(path_outputs) else os.path.abspath(os.path.join(os.path.dirname(__file__), path_outputs))

os.makedirs(temp_outputs_path, exist_ok=True)

default_base_model_name = 'sd_xl_base_1.0_0.9vae.safetensors'
default_refiner_model_name = 'sd_xl_refiner_1.0_0.9vae.safetensors'
default_lora_name = 'sd_xl_offset_example-lora_1.0.safetensors'
default_clip_vision_name = 'clip_vision_g.safetensors'
default_controlnet_canny_name = 'control-lora-canny-rank128.safetensors'
default_controlnet_depth_name = 'control-lora-depth-rank128.safetensors'
default_lora_weight = 0.5

model_filenames = []
lora_filenames = []
canny_filenames = []
depth_filenames = []


def get_files_from_folder(folder_path, exensions=None, name_filter=None):
    if not os.path.isdir(folder_path):
        raise ValueError("Folder path is not a valid directory.")

    filenames = []

    for root, dirs, files in os.walk(folder_path):
        relative_path = os.path.relpath(root, folder_path)
        if relative_path == ".":
            relative_path = ""
        for filename in files:
            _, file_extension = os.path.splitext(filename)
            if (exensions == None or file_extension.lower() in exensions) and (name_filter == None or name_filter in _):
                path = os.path.join(relative_path, filename)
                filenames.append(path)

    return sorted(filenames, key=lambda x: -1 if os.sep in x else 1)


def get_model_filenames(folder_path, name_filter=None):
    return get_files_from_folder(folder_path, ['.pth', '.ckpt', '.bin', '.safetensors'], name_filter)


def update_all_model_names():
    global model_filenames, lora_filenames, canny_filenames, depth_filenames
    model_filenames = get_model_filenames(modelfile_path)
    lora_filenames = get_model_filenames(lorafile_path)
    canny_filenames = get_model_filenames(controlnet_path, 'control-lora-canny')
    depth_filenames = get_model_filenames(controlnet_path, 'control-lora-depth')
    return


update_all_model_names()
