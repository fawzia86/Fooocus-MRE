import gradio as gr
import random
import time
import shared
import argparse
import modules.path
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import json

from modules.settings import load_settings
from modules.resolutions import get_resolution_string, resolutions
from modules.sdxl_styles import style_keys
from collections.abc import Mapping
from PIL import Image


def generate_clicked(*args):
    yield gr.update(interactive=False), \
        gr.update(visible=True, value=modules.html.make_progress_html(1, 'Processing text encoding ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=False), \
        gr.update(), \
        gr.update(value=None), \
        gr.update()

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
                    gr.update(), \
                    gr.update(), \
                    gr.update()
            if flag == 'results':
                yield gr.update(interactive=True), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=True), \
                    gr.update(value=product), \
                    gr.update(), \
                    gr.update()
            if flag == 'metadatas':
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value=product), gr.update(selected=1)
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
    if 'width' in metadata and 'height' in metadata:
        ctrls[4] = get_resolution_string(metadata['width'], metadata['height'])
    elif 'resolution' in metadata:
        ctrls[4] = metadata['resolution']
    # image_number
    if 'seed' in metadata:
        ctrls[6] = metadata['seed']
        ctrls[32] = False
    if 'sharpness' in metadata:
        ctrls[7] = metadata['sharpness']
    if 'sampler_name' in metadata:
        ctrls[8] = metadata['sampler_name']
    elif 'sampler' in metadata:
        ctrls[8] = metadata['sampler']
    if 'scheduler' in metadata:
        ctrls[9] = metadata['scheduler']
    if 'steps' in metadata:
        ctrls[10] = metadata['steps']
        if ctrls[10] == constants.STEPS_SPEED:
            ctrls[3] = 'Speed'
        elif ctrls[10] == constants.STEPS_QUALITY:
            ctrls[3] = 'Quality'
        else:
            ctrls[3] = 'Custom'
    if 'switch' in metadata:
        ctrls[11] = round(metadata['switch'] / ctrls[10], 2)
        if ctrls[11] != round(constants.SWITCH_SPEED / constants.STEPS_SPEED, 2):
            ctrls[3] = 'Custom'
    if 'cfg' in metadata:
        ctrls[12] = metadata['cfg']
    if 'base_model' in metadata:
        ctrls[13] = metadata['base_model']
    elif 'base_model_name' in metadata:
        ctrls[13] = metadata['base_model_name']
    if 'refiner_model' in metadata:
        ctrls[14] = metadata['refiner_model']
    elif 'refiner_model_name' in metadata:
        ctrls[14] = metadata['refiner_model_name']
    if 'base_clip_skip' in metadata:
        ctrls[15] = metadata['base_clip_skip']
    if 'refiner_clip_skip' in metadata:
        ctrls[16] = metadata['refiner_clip_skip']
    if 'l1' in metadata:
        ctrls[17] = metadata['l1']
    if 'w1' in metadata:
        ctrls[18] = metadata['w1']
    if 'l2' in metadata:
        ctrls[19] = metadata['l2']
    if 'w2' in metadata:
        ctrls[20] = metadata['w2']
    if 'l3' in metadata:
        ctrls[21] = metadata['l3']
    if 'w3' in metadata:
        ctrls[22] = metadata['w3']
    if 'l4' in metadata:
        ctrls[23] = metadata['l4']
    if 'w4' in metadata:
        ctrls[24] = metadata['w4']
    if 'l5' in metadata:
        ctrls[25] = metadata['l5']
    if 'w5' in metadata:
        ctrls[26] = metadata['w5']
    # save_metadata_json
    # save_metadata_png
    if 'img2img' in metadata:
        ctrls[29] = metadata['img2img']
        if 'start_step' in metadata:
            if ctrls[3] == 'Speed':
                ctrls[30] = round(metadata['start_step'] / constants.STEPS_SPEED, 2)
            elif ctrls[3] == 'Quality':
                ctrls[30] = round(metadata['start_step'] / constants.STEPS_QUALITY, 2)
            else:
                ctrls[30] = round(metadata['start_step'] / ctrls[10], 2)
        if 'denoise' in metadata:
            ctrls[31] = metadata['denoise']
    if 'revision' in metadata:
        ctrls[32] = metadata['revision']
    if 'zero_out' in metadata:
        ctrls[33] = metadata['zero_out']
    if 'revision_weight' in metadata:
        ctrls[34] = metadata['revision_weight']
    if 'revision_noise' in metadata:
        ctrls[35] = metadata['revision_noise']
    # seed_random
    return ctrls    


def load_prompt_handler(_file, *args):
    ctrls=list(args)
    path = _file.name
    if path.endswith('.json'):
        with open(path, encoding='utf-8') as json_file:
            try:
                json_obj = json.load(json_file)
                metadata_to_ctrls(json_obj, ctrls)
            except Exception as e:
                print(e)
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
                except Exception as e:
                    print(e)
    return ctrls


def load_images_handler(files):
    return gr.update(value=True), list(map(lambda x: x.name, files)), gr.update(selected=0)


def output_to_input_handler(gallery):
    if len(gallery) == 0:
        return gr.update(value=False), [], gr.update()
    else:
        return gr.update(value=True), list(map(lambda x: x['name'], gallery)), gr.update(selected=0)


settings = load_settings()

shared.gradio_root = gr.Blocks(title=fooocus_version.full_version, css=modules.html.css).queue()
with shared.gradio_root:
    with gr.Row():
        with gr.Column():
            progress_window = gr.Image(label='Preview', show_label=True, height=640, visible=False)
            progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False, elem_id='progress-bar', elem_classes='progress-bar')
            with gr.Column() as gallery_holder:
                with gr.Tabs(selected=1) as gallery_tabs:
                    with gr.Tab(label='Input', id=0):
                        input_gallery = gr.Gallery(label='Input', show_label=False, object_fit='contain', height=720, visible=True)
                    with gr.Tab(label='Output', id=1):
                        output_gallery = gr.Gallery(label='Output', show_label=False, object_fit='contain', height=720, visible=True)
            with gr.Row(elem_classes='type_row'):
                with gr.Column(scale=0.85):
                    prompt = gr.Textbox(show_label=False, placeholder='Type prompt here.', container=False, autofocus=True, elem_classes='type_row', lines=1024, value=settings['prompt'])
                with gr.Column(scale=0.15, min_width=0):
                    with gr.Row():
                        img2img_mode = gr.Checkbox(label='Image-2-Image', value=settings['img2img_mode'], elem_classes='type_small_row')
                    with gr.Row():
                        run_button = gr.Button(label='Generate', value='Generate', elem_classes='type_small_row')
            with gr.Row():
                advanced_checkbox = gr.Checkbox(label='Advanced', value=settings['advanced_mode'], container=False)

            def verify_input(img2img, gallery_in, gallery_out):
                if img2img and len(gallery_in) == 0:
                    if len(gallery_out) == 0:
                        gr.Warning('Image-2-Image: disabled (no images available)')
                        return gr.update(value=False), gr.update(), gr.update()
                    else:
                        gr.Info('Image-2-Image: imported output as input')
                        return gr.update(), list(map(lambda x: x['name'], gallery_out)), gr.update()
                else:
                    return gr.update(), gr.update(), gr.update()

        with gr.Column(scale=0.5, visible=settings['advanced_mode']) as advanced_column:
            with gr.Tab(label='Settings'):
                performance = gr.Radio(label='Performance', choices=['Speed', 'Quality', 'Custom'], value=settings['performance'])
                with gr.Row():
                    custom_steps = gr.Slider(label='Custom Steps', minimum=10, maximum=200, step=1, value=settings['custom_steps'], visible=settings['performance'] == 'Custom')
                    custom_switch = gr.Slider(label='Custom Switch', minimum=0.2, maximum=1.0, step=0.01, value=settings['custom_switch'], visible=settings['performance'] == 'Custom')
                resolution = gr.Dropdown(label='Resolution (width × height)', choices=list(resolutions.keys()), value=settings['resolution'])
                style_selection = gr.Dropdown(label='Style', choices=style_keys, value=settings['style'])
                image_number = gr.Slider(label='Image Number', minimum=1, maximum=32, step=1, value=settings['image_number'])
                negative_prompt = gr.Textbox(label='Negative Prompt', show_label=True, placeholder="Type prompt here.", value=settings['negative_prompt'])
                seed_random = gr.Checkbox(label='Random', value=settings['seed_random'])
                image_seed = gr.Number(label='Seed', value=settings['seed'], precision=0, visible=not settings['seed_random'])
                with gr.Row():
                    load_prompt_button = gr.UploadButton(label='Load Prompt', file_count='single', file_types=['.json', '.png'], elem_classes='type_small_row', min_width=0)
                    load_images_button = gr.UploadButton(label='Load Image(s)', file_count='multiple', file_types=["image"], elem_classes='type_small_row', min_width=0)
                    output_to_input_button = gr.Button(label='Output to Input', value='Output to Input', elem_classes='type_small_row', min_width=0)

                def random_checked(r):
                    return gr.update(visible=not r)

                def refresh_seed(r, s):
                    if r or not isinstance(s, int) or s < 0 or s > 2**63 - 1:
                        return random.randint(0, 2**63 - 1)
                    else:
                        return s

                seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed])

                def performance_changed(value):
                    return gr.update(visible=value == 'Custom'), gr.update(visible=value == 'Custom')

                performance.change(fn=performance_changed, inputs=[performance], outputs=[custom_steps, custom_switch])
                load_images_button.upload(fn=load_images_handler, inputs=[load_images_button], outputs=[img2img_mode, input_gallery, gallery_tabs])
                output_to_input_button.click(output_to_input_handler, inputs=output_gallery, outputs=[img2img_mode, input_gallery, gallery_tabs])

            with gr.Tab(label='Image-2-Image'):
                with gr.Row():
                    revision_mode = gr.Checkbox(label='Revision', value=settings['revision_mode'], elem_classes='type_small_row')
                    zero_out = gr.Checkbox(label='Zero Out Prompts', value=settings['zero_out'], elem_classes='type_small_row')
                revision_weight = gr.Slider(label='Revision Weight', minimum=-2, maximum=2, step=0.01, value=settings['revision_weight'])
                revision_noise = gr.Slider(label='Revision Noise', minimum=0, maximum=1, step=0.01, value=settings['revision_noise'])
                revision_ctrls = [revision_mode, zero_out, revision_weight, revision_noise]
                img2img_start_step = gr.Slider(label='Image-2-Image Start Step', minimum=0.0, maximum=0.8, step=0.01, value=settings['img2img_start_step'])
                img2img_denoise = gr.Slider(label='Image-2-Image Denoise', minimum=0.2, maximum=1.0, step=0.01, value=settings['img2img_denoise'])

            with gr.Tab(label='Models'):
                with gr.Row():
                    base_model = gr.Dropdown(label='SDXL Base Model', choices=modules.path.model_filenames, value=settings['base_model'], show_label=True)
                    refiner_model = gr.Dropdown(label='SDXL Refiner', choices=['None'] + modules.path.model_filenames, value=settings['refiner_model'], show_label=True)
                with gr.Accordion(label='LoRAs', open=True):
                    lora_ctrls = []
                    for i in range(5):
                        with gr.Row():
                            lora_model = gr.Dropdown(label=f'SDXL LoRA {i+1}', choices=['None'] + modules.path.lora_filenames, value=settings[f'lora_{i+1}_model'])
                            lora_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.01, value=settings[f'lora_{i+1}_weight'])
                            lora_ctrls += [lora_model, lora_weight]
                with gr.Row():
                    model_refresh = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files', variant='secondary', elem_classes='refresh_button')

            with gr.Tab(label='Advanced'):
                cfg = gr.Slider(label='CFG', minimum=1.0, maximum=20.0, step=0.1, value=settings['cfg'])
                base_clip_skip = gr.Slider(label='Base CLIP Skip', minimum=-10, maximum=-1, step=1, value=settings['base_clip_skip'])
                refiner_clip_skip = gr.Slider(label='Refiner CLIP Skip', minimum=-10, maximum=-1, step=1, value=settings['refiner_clip_skip'])
                sampler_name = gr.Dropdown(label='Sampler', choices=['dpmpp_2m_sde_gpu', 'dpmpp_2m_sde', 'dpmpp_3m_sde_gpu', 'dpmpp_3m_sde',
                    'dpmpp_sde_gpu', 'dpmpp_sde', 'dpmpp_2s_ancestral', 'euler', 'euler_ancestral', 'heun', 'dpm_2', 'dpm_2_ancestral'], value=settings['sampler'])
                scheduler = gr.Dropdown(label='Scheduler', choices=['karras', 'exponential', 'simple', 'ddim_uniform'], value=settings['scheduler'])
                sharpness = gr.Slider(label='Sampling Sharpness', minimum=0.0, maximum=40.0, step=0.01, value=settings['sharpness'])
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
                with gr.Row():
                    save_metadata_json = gr.Checkbox(label='Save Metadata in JSON', value=settings['save_metadata_json'])
                    save_metadata_png = gr.Checkbox(label='Save Metadata in PNG', value=settings['save_metadata_png'])
                metadata_viewer = gr.JSON(label='Metadata')

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, advanced_column)
        ctrls = [
            prompt, negative_prompt, style_selection,
            performance, resolution, image_number, image_seed, sharpness, sampler_name, scheduler,
            custom_steps, custom_switch, cfg
        ]
        ctrls += [base_model, refiner_model, base_clip_skip, refiner_clip_skip] + lora_ctrls \
            + [save_metadata_json, save_metadata_png, img2img_mode, img2img_start_step, img2img_denoise] + revision_ctrls
        load_prompt_button.upload(fn=load_prompt_handler, inputs=[load_prompt_button] + ctrls + [seed_random], outputs=ctrls + [seed_random])
        run_button.click(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
            .then(fn=verify_input, inputs=[img2img_mode, input_gallery, output_gallery], outputs=[img2img_mode, input_gallery, output_gallery]) \
            .then(fn=generate_clicked, inputs=ctrls + [input_gallery], outputs=[run_button, progress_html, progress_window, gallery_holder, output_gallery, metadata_viewer, gallery_tabs])

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=None, help="Set the listen port.")
parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")
parser.add_argument("--listen", type=str, default=None, metavar="IP", nargs="?", const="0.0.0.0", help="Set the listen interface.")
args = parser.parse_args()
shared.gradio_root.launch(inbrowser=True, server_name=args.listen, server_port=args.port, share=args.share)
