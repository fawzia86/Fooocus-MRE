import os
import json

from os.path import exists
from modules.path import styles_path, get_files_from_folder


base_styles = [
    {
        "name": "None",
        "prompt": "{prompt}",
        "negative_prompt": ""
    },
    {
        "name": "cinematic-default",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
    }
]


default_styles_files = ['sdxl_styles_sai.json', 'sdxl_styles_twri.json', 'sdxl_styles_diva.json', 'sdxl_styles_mre.json']


def styles_list_to_styles_dict(styles_list=None, base_dict=None):
    styles_dict = {} if base_dict == None else base_dict
    if isinstance(styles_list, list) and len(styles_list) > 0:
        for entry in styles_list:
            name, prompt, negative_prompt = entry['name'], entry['prompt'], entry['negative_prompt']
            if name not in styles_dict:
                styles_dict |=  {name: (prompt, negative_prompt)}
    return styles_dict


def load_styles(filename=None, base_dict=None):
    styles_dict = {} if base_dict == None else base_dict
    full_path = os.path.join(styles_path, filename) if filename != None else None
    if full_path != None and exists(full_path):
        with open(full_path, encoding='utf-8') as styles_file:
            try:
                styles_obj = json.load(styles_file)
                styles_list_to_styles_dict(styles_obj, styles_dict)
            except Exception as e:
                print('load_styles, e: ' + str(e))
            finally:
                styles_file.close()
    return styles_dict


styles = styles_list_to_styles_dict(base_styles)
for styles_file in default_styles_files:
    styles = load_styles(styles_file, styles)


all_styles_files = get_files_from_folder(styles_path, ['.json'])
for styles_file in all_styles_files:
    if styles_file not in default_styles_files:
        styles = load_styles(styles_file, styles)


default_style = styles['None']
style_keys = list(styles.keys())


def apply_style_positive(style, txt):
    p, n = styles.get(style, default_style)
    ps = p.split('{prompt}')
    if len(ps) != 2:
        return txt, ''
    return ps[0] + txt, ps[1]


def apply_style_negative(style, txt):
    p, n = styles.get(style, default_style)
    if n == '':
        return txt
    elif txt == '':
        return n
    else:
        return n + ', ' + txt
