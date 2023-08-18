import gradio as gr
import random
import time
import shared
import argparse
import modules.path
import fooocus_version
import modules.html
import modules.async_worker as worker
import json

from modules.sdxl_styles import style_keys, aspect_ratios
from collections.abc import Mapping
from PIL import Image
from os.path import exists

def generate_clicked(*args):
    yield gr.update(interactive=False), \
        gr.update(visible=True, value=modules.html.make_progress_html(1, 'Processing text encoding ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=False), \
        gr.update(value=None)

    worker.buffer.append(list(args))
    finished = False

    while not finished:
        time.sleep(0.01)
        if len(worker.outputs) > 0:
            flag, product = worker.outputs.pop(0)
            if flag == 'preview':
                percentage, title, image = product
                yield gr.update(interactive=False), \
                    gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(), \
                    gr.update(visible=False), \
                    gr.update()
            if flag == 'results':
                yield gr.update(interactive=True), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=True, value=product), \
                    gr.update()
            if flag == 'metadatas':
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value=product)
                finished = True
    return


def metadata_to_ctrls(metadata, ctrls):
    if not isinstance(metadata, Mapping):
        return ctrls

    if 'prompt' in metadata:
        ctrls[0] = metadata['prompt']
    if 'negative_prompt' in metadata:
        ctrls[1] = metadata['negative_prompt']
    if 'style' in metadata:
        ctrls[2] = metadata['style']
    if 'performance' in metadata:
        ctrls[3] = metadata['performance']
    if 'resolution' in metadata:
        ctrls[4] = metadata['resolution']
    elif 'width' in metadata and 'height' in metadata:
        ctrls[4] = str(metadata['width'])+'×'+str(metadata['height'])
    # image_number
    if 'seed' in metadata:
        ctrls[6] = metadata['seed']
        ctrls[31] = False
    if 'sharpness' in metadata:
        ctrls[7] = metadata['sharpness']
    if 'sampler_name' in metadata:
        ctrls[8] = metadata['sampler_name']
    elif 'sampler' in metadata:
        ctrls[8] = metadata['sampler']
    if 'scheduler' in metadata:
        ctrls[9] = metadata['scheduler']
    if 'steps' in metadata:
        if ctrls[3] == 'Speed':
            ctrls[10] = metadata['steps']
        else:
            ctrls[12] = metadata['steps']
    if 'switch' in metadata:
        if ctrls[3] == 'Speed':
           ctrls[11] = round(metadata['switch'] / ctrls[10], 2)
        else:
            ctrls[13] = round(metadata['switch'] / ctrls[12], 2)
    if 'cfg' in metadata:
        ctrls[14] = metadata['cfg']
    if 'base_model' in metadata:
        ctrls[15] = metadata['base_model']
    elif 'base_model_name' in metadata:
        ctrls[15] = metadata['base_model_name']
    if 'refiner_model' in metadata:
        ctrls[16] = metadata['refiner_model']
    elif 'refiner_model_name' in metadata:
        ctrls[16] = metadata['refiner_model_name']
    if 'base_clip_skip' in metadata:
        ctrls[17] = metadata['base_clip_skip']
    if 'refiner_clip_skip' in metadata:
        ctrls[18] = metadata['refiner_clip_skip']
    if 'l1' in metadata:
        ctrls[19] = metadata['l1']
    if 'w1' in metadata:
        ctrls[20] = metadata['w1']
    if 'l2' in metadata:
        ctrls[21] = metadata['l2']
    if 'w2' in metadata:
        ctrls[22] = metadata['w2']
    if 'l3' in metadata:
        ctrls[23] = metadata['l3']
    if 'w3' in metadata:
        ctrls[24] = metadata['w3']
    if 'l4' in metadata:
        ctrls[25] = metadata['l4']
    if 'w4' in metadata:
        ctrls[26] = metadata['w4']
    if 'l5' in metadata:
        ctrls[27] = metadata['l5']
    if 'w5' in metadata:
        ctrls[28] = metadata['w5']
    # save_metadata_json
    # save_metadata_png
    # seed_random

    return ctrls    


def load_handler(files, *args):
    ctrls=list(args)
    if len(files) > 0:
        path = files[0].name
        if path.endswith('.json'):
            with open(path) as json_file:
                try:
                    json_obj = json.load(json_file)
                    metadata_to_ctrls(json_obj, ctrls)
                except Exception:
                    pass
                finally:
                    json_file.close()
        elif path.endswith('.png'):
            with open(path, 'rb') as png_file:
                image = Image.open(png_file)
                png_file.close()
                if 'Comment' in image.info:
                    try:
                        metadata = json.loads(image.info['Comment'])
                        metadata_to_ctrls(metadata, ctrls)
                    except Exception:
                        pass
    return ctrls


def load_settings():
    advanced_mode = False
    image_number = 2
    save_metadata_json = False
    save_metadata_png = False
    seed_random = True

    if exists('settings.json'):
        with open('settings.json') as settings_file:
            try:
                stored_settings = json.load(settings_file)
                if 'advanced_mode' in stored_settings:
                    advanced_mode = stored_settings['advanced_mode']
                if 'image_number' in stored_settings:
                    image_number = stored_settings['image_number']
                if 'save_metadata_json' in stored_settings:
                    save_metadata_json = stored_settings['save_metadata_json']
                if 'save_metadata_png' in stored_settings:
                    save_metadata_png = stored_settings['save_metadata_png']
                if 'seed_random' in stored_settings:
                    seed_random = stored_settings['seed_random']
            except Exception:
                pass
            finally:
                settings_file.close()

    return advanced_mode, image_number, save_metadata_json, save_metadata_png, seed_random


advanced_mode_value, image_number_value, save_metadata_json_value, save_metadata_png_value, seed_random_value = load_settings()

shared.gradio_root = gr.Blocks(title=fooocus_version.full_version, css=modules.html.css).queue()
with shared.gradio_root:
    with gr.Row():
        with gr.Column():
            progress_window = gr.Image(label='Preview', show_label=True, height=640, visible=False)
            progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False, elem_id='progress-bar', elem_classes='progress-bar')
            gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', height=720, visible=True)
            with gr.Row(elem_classes='type_row'):
                with gr.Column(scale=0.7):
                    prompt = gr.Textbox(show_label=False, placeholder='Type prompt here.', container=False, autofocus=True, elem_classes='type_row', lines=1024)
                with gr.Column(scale=0.15, min_width=0):
                    load_button = gr.UploadButton(label='Load Prompt', elem_classes='type_row', file_count=1, file_types=['.json', '.png'])
                with gr.Column(scale=0.15, min_width=0):
                    run_button = gr.Button(label='Generate', value='Generate', elem_classes='type_row')
            with gr.Row():
                advanced_checkbox = gr.Checkbox(label='Advanced', value=advanced_mode_value, container=False)
        with gr.Column(scale=0.52, visible=advanced_mode_value) as right_col:
            with gr.Tab(label='Setting'):
                performance_selection = gr.Radio(label='Performance', choices=['Speed', 'Quality'], value='Speed')
                aspect_ratios_selection = gr.Radio(label='Aspect Ratios (width × height)', choices=list(aspect_ratios.keys()), value='1152×896')
                image_number = gr.Slider(label='Image Number', minimum=1, maximum=32, step=1, value=image_number_value)
                negative_prompt = gr.Textbox(label='Negative Prompt', show_label=True, placeholder="Type prompt here.")
                seed_random = gr.Checkbox(label='Random', value=seed_random_value)
                image_seed = gr.Number(label='Seed', value=0, precision=0, visible=not seed_random_value)

                def random_checked(r):
                    return gr.update(visible=not r)

                def refresh_seed(r, s):
                    if r or not isinstance(s, int) or s < 0 or s > 2**63 - 1:
                        return random.randint(0, 2**63 - 1)
                    else:
                        return s

                seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed])

            with gr.Tab(label='Style'):
                style_selection = gr.Radio(show_label=False, container=True,
                                          choices=style_keys, value='cinematic-default')
            with gr.Tab(label='Models'):
                with gr.Row():
                    base_model = gr.Dropdown(label='SDXL Base Model', choices=modules.path.model_filenames, value=modules.path.default_base_model_name, show_label=True)
                    refiner_model = gr.Dropdown(label='SDXL Refiner', choices=['None'] + modules.path.model_filenames, value=modules.path.default_refiner_model_name, show_label=True)
                with gr.Accordion(label='LoRAs', open=True):
                    lora_ctrls = []
                    for i in range(5):
                        with gr.Row():
                            lora_model = gr.Dropdown(label=f'SDXL LoRA {i+1}', choices=['None'] + modules.path.lora_filenames, value=modules.path.default_lora_name if i == 0 else 'None')
                            lora_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.01, value=modules.path.default_lora_weight)
                            lora_ctrls += [lora_model, lora_weight]
                with gr.Row():
                    model_refresh = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files', variant='secondary', elem_classes='refresh_button')
            with gr.Tab(label='Advanced'):
                with gr.Row():
                    save_metadata_json = gr.Checkbox(label='Save Metadata in JSON', value=save_metadata_json_value)
                    save_metadata_png = gr.Checkbox(label='Save Metadata in PNG', value=save_metadata_png_value)
                cfg = gr.Slider(label='CFG', minimum=1.0, maximum=20.0, step=0.1, value=7.0)
                base_clip_skip = gr.Slider(label='Base CLIP Skip', minimum=-10, maximum=-1, step=1, value=-2)
                refiner_clip_skip = gr.Slider(label='Refiner CLIP Skip', minimum=-10, maximum=-1, step=1, value=-2)
                sampler_name = gr.Dropdown(label='Sampler', choices=['dpmpp_2m_sde_gpu', 'dpmpp_2m_sde', 'dpmpp_3m_sde_gpu', 'dpmpp_3m_sde',
                    'dpmpp_sde_gpu', 'dpmpp_sde', 'dpmpp_2s_ancestral', 'euler', 'euler_ancestral', 'heun', 'dpm_2', 'dpm_2_ancestral'], value='dpmpp_2m_sde_gpu')
                scheduler = gr.Dropdown(label='Scheduler', choices=['karras', 'exponential', 'simple', 'ddim_uniform'], value='karras')
                sampler_steps_speed = gr.Slider(label='Sampler Steps (Speed)', minimum=10, maximum=100, step=1, value=30)
                switch_step_speed = gr.Slider(label='Switch Step (Speed)', minimum=0.5, maximum=1.0, step=0.01, value=0.67)
                sampler_steps_quality = gr.Slider(label='Sampler Steps (Quality)', minimum=20, maximum=200, step=1, value=60)
                switch_step_quality = gr.Slider(label='Switch Step (Quality)', minimum=0.5, maximum=1.0, step=0.01, value=0.67)
                sharpness = gr.Slider(label='Sampling Sharpness', minimum=0.0, maximum=40.0, step=0.01, value=2.0)
                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/117">\U0001F4D4 Document</a>')

                def model_refresh_clicked():
                    modules.path.update_all_model_names()
                    results = []
                    results += [gr.update(choices=modules.path.model_filenames), gr.update(choices=['None'] + modules.path.model_filenames)]
                    for i in range(5):
                        results += [gr.update(choices=['None'] + modules.path.lora_filenames), gr.update()]
                    return results

                model_refresh.click(model_refresh_clicked, [], [base_model, refiner_model] + lora_ctrls)
            with gr.Tab(label='Metadata'):
                metadata_viewer = gr.JSON(label='Metadata')

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, right_col)
        ctrls = [
            prompt, negative_prompt, style_selection,
            performance_selection, aspect_ratios_selection, image_number, image_seed, sharpness, sampler_name, scheduler,
            sampler_steps_speed, switch_step_speed, sampler_steps_quality, switch_step_quality, cfg
        ]
        ctrls += [base_model, refiner_model, base_clip_skip, refiner_clip_skip] + lora_ctrls + [save_metadata_json, save_metadata_png]
        run_button.click(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed)\
            .then(fn=generate_clicked, inputs=ctrls, outputs=[run_button, progress_html, progress_window, gallery, metadata_viewer])
        load_button.upload(fn=load_handler, inputs=[load_button] + ctrls + [seed_random], outputs=ctrls + [seed_random])

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=None, help="Set the listen port.")
parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")
parser.add_argument("--listen", type=str, default=None, metavar="IP", nargs="?", const="0.0.0.0", help="Set the listen interface.")
args = parser.parse_args()
shared.gradio_root.launch(inbrowser=True, server_name=args.listen, server_port=args.port, share=args.share)
