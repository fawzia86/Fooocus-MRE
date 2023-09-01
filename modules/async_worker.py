import os
import threading
import json
import torch
import numpy as np

import modules.core as core
import modules.constants as constants

from PIL import Image, ImageOps
from modules.resolutions import annotate_resolution_string


buffer = []
outputs = []


def get_image(path):
    image = None
    with open(path, 'rb') as image_file:
        pil_image = Image.open(image_file)
        image = ImageOps.exif_transpose(pil_image)
        image_file.close()
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        image = core.upscale(image)
    return image


def worker():
    global buffer, outputs

    import time
    import shared
    import random
    import modules.default_pipeline as pipeline
    import modules.path
    import modules.patch
    import fooocus_version

    from modules.resolutions import get_resolution_string, resolutions
    from modules.sdxl_styles import apply_style
    from modules.private_logger import log

    try:
        async_gradio_app = shared.gradio_root
        flag = f'''App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'''
        if async_gradio_app.share:
            flag += f''' or {async_gradio_app.share_url}'''
        print(flag)
    except Exception as e:
        print(e)

    def handler(task):
        prompt, negative_prompt, style, performance, resolution, image_number, image_seed, \
        sharpness, sampler_name, scheduler, custom_steps, custom_switch, cfg, \
        base_model_name, refiner_model_name, base_clip_skip, refiner_clip_skip, \
        l1, w1, l2, w2, l3, w3, l4, w4, l5, w5, save_metadata_json, save_metadata_image, \
        img2img_mode, img2img_start_step, img2img_denoise, \
        revision_mode, zero_out_positive, zero_out_negative, revision_strength_1, revision_strength_2, \
        revision_strength_3, revision_strength_4, same_seed_for_all, output_format, \
        control_lora_canny, canny_edge_low, canny_edge_high, canny_start, canny_stop, canny_strength, canny_model, \
        control_lora_depth, depth_start, depth_stop, depth_strength, depth_model, \
        input_gallery, revision_gallery, keep_input_names = task

        loras = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]

        modules.patch.sharpness = sharpness

        input_gallery_size = len(input_gallery)
        if input_gallery_size == 0:
            img2img_mode = False
            input_image_path = None
            control_lora_canny = False
            control_lora_depth = False

        revision_gallery_size = len(revision_gallery)
        if revision_gallery_size == 0:
            revision_mode = False

        pipeline.refresh_base_model(base_model_name)
        pipeline.refresh_refiner_model(refiner_model_name)
        pipeline.refresh_loras(loras)
        if revision_mode:
            pipeline.refresh_clip_vision()
        if control_lora_canny:
            pipeline.refresh_controlnet_canny(canny_model)
        if control_lora_depth:
            pipeline.refresh_controlnet_depth(depth_model)

        p_txt, n_txt = apply_style(style, prompt, negative_prompt)

        if performance == 'Speed':
            steps = constants.STEPS_SPEED
            switch = constants.SWITCH_SPEED
        elif performance == 'Quality':
            steps = constants.STEPS_QUALITY
            switch = constants.SWITCH_QUALITY
        else:
            steps = custom_steps
            switch = round(custom_steps * custom_switch)

        if resolution not in resolutions:
            resolution = annotate_resolution_string(resolution)
        width, height = resolutions[resolution]

        results = []
        metadata_strings = []

        try:
            seed = int(image_seed) 
        except Exception as e:
            seed = -1

        if not isinstance(seed, int) or seed < constants.MIN_SEED or seed > constants.MAX_SEED:
            seed = random.randint(constants.MIN_SEED, constants.MAX_SEED)

        all_steps = steps * image_number

        def callback(step, x0, x, total_steps, y):
            done_steps = i * steps + step
            outputs.append(['preview', (
                int(100.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {i}-th Sampling',
                y)])

        clip_vision_outputs = []
        if revision_mode:
            revision_images_paths = list(map(lambda x: x['name'], revision_gallery))
            revision_images_filenames = list(map(lambda path: os.path.basename(path), revision_images_paths))
            revision_strengths = [revision_strength_1, revision_strength_2, revision_strength_3, revision_strength_4]
            for i in range(revision_gallery_size):
                print(f'Revision for image {i+1} started')
                if revision_strengths[i % 4] != 0:
                    revision_image = get_image(revision_images_paths[i])
                    clip_vision_output = core.encode_clip_vision(pipeline.clip_vision, revision_image)
                    clip_vision_outputs.append(clip_vision_output)
                print(f'Revision for image {i+1} finished')
        else:
            revision_images_paths = []
            revision_images_filenames = []
            revision_strengths = []

        for i in range(image_number):
            if img2img_mode or control_lora_canny or control_lora_depth:
                input_gallery_entry = input_gallery[i % input_gallery_size]
                input_image_path = input_gallery_entry['name']
                input_image_filename = None if input_image_path == None else os.path.basename(input_image_path)
            else:
                input_image_path = None
                input_image_filename = None
                keep_input_names = None

            if img2img_mode:
                start_step = round(steps * img2img_start_step)
                denoise = img2img_denoise
            else:
                start_step = 0
                denoise = None

            input_image = None
            if input_image_path != None:
                input_image = get_image(input_image_path)

            execution_start_time = time.perf_counter()
            imgs = pipeline.process(p_txt, n_txt, steps, switch, width, height, seed, sampler_name, scheduler,
                cfg, base_clip_skip, refiner_clip_skip, img2img_mode, input_image, start_step, denoise,
                revision_mode, clip_vision_outputs, zero_out_positive, zero_out_negative, revision_strengths,
                control_lora_canny, canny_edge_low, canny_edge_high, canny_start, canny_stop, canny_strength,
                control_lora_depth, depth_start, depth_stop, depth_strength, callback=callback)
            execution_time = time.perf_counter() - execution_start_time
            print(f"Prompt executed in {execution_time:.2f} seconds")

            metadata = {
                'prompt': prompt, 'negative_prompt': negative_prompt, 'style': style,
                'seed': seed, 'width': width, 'height': height, 'p_txt': p_txt, 'n_txt': n_txt,
                'sampler': sampler_name, 'scheduler': scheduler, 'performance': performance,
                'steps': steps, 'switch': switch, 'sharpness': sharpness, 'cfg': cfg,
                'base_clip_skip': base_clip_skip, 'refiner_clip_skip': refiner_clip_skip,
                'base_model': base_model_name, 'refiner_model': refiner_model_name,
                'l1': l1, 'w1': w1, 'l2': l2, 'w2': w2, 'l3': l3, 'w3': w3,
                'l4': l4, 'w4': w4, 'l5': l5, 'w5': w5, 'img2img': img2img_mode, 'revision': revision_mode,
                'zero_out_positive': zero_out_positive, 'zero_out_negative': zero_out_negative,
                'control_lora_canny': control_lora_canny, 'control_lora_depth': control_lora_depth
            }
            if img2img_mode:
                metadata |= {
                    'start_step': start_step, 'denoise': denoise, 'input_image': input_image_filename
                }
            if revision_mode:
                metadata |= {
                    'revision_strength_1': revision_strength_1, 'revision_strength_2': revision_strength_2,
                    'revision_strength_3': revision_strength_3, 'revision_strength_4': revision_strength_4,
                    'revision_images': revision_images_filenames
                }
            if control_lora_canny:
                metadata |= {
                    'canny_edge_low': canny_edge_low, 'canny_edge_high': canny_edge_high, 'canny_start': canny_start,
                    'canny_stop': canny_stop, 'canny_strength': canny_strength, 'canny_model': canny_model, 'canny_input': input_image_filename
                }
            if control_lora_depth:
                metadata |= {
                    'depth_start': depth_start, 'depth_stop': depth_stop, 'depth_strength': depth_strength, 'depth_model': depth_model, 'depth_input': input_image_filename
                }
            metadata |= { 'software': fooocus_version.full_version }

            metadata_string = json.dumps(metadata, ensure_ascii=False)
            metadata_strings.append(metadata_string)

            for x in imgs:
                d = [
                    ('Prompt', prompt),
                    ('Negative Prompt', negative_prompt),
                    ('Style', style),
                    ('Seed', seed),
                    ('Resolution', get_resolution_string(width, height)),
                    ('Performance', (performance, steps, switch)),
                    ('Sampler & Scheduler', (sampler_name, scheduler)),
                    ('Sharpness', sharpness),
                    ('CFG & CLIP Skips', (cfg, base_clip_skip, refiner_clip_skip)),
                    ('Base Model', base_model_name),
                    ('Refiner Model', refiner_model_name),
                    ('Image-2-Image', (img2img_mode, start_step, denoise, input_image_filename) if img2img_mode else (img2img_mode)),
                    ('Revision', (revision_mode, revision_strength_1, revision_strength_2, revision_strength_3,
                        revision_strength_4, revision_images_filenames) if revision_mode else (revision_mode)),
                    ('Zero Out Prompts', (zero_out_positive, zero_out_negative)),
                    ('Canny', (control_lora_canny, canny_edge_low, canny_edge_high, canny_start, canny_stop,
                        canny_strength, canny_model, input_image_filename) if control_lora_canny else (control_lora_canny)),
                    ('Depth', (control_lora_depth, depth_start, depth_stop, depth_strength, depth_model, input_image_filename) if control_lora_depth else (control_lora_depth))
                ]
                for n, w in loras:
                    if n != 'None':
                        d.append((f'LoRA [{n}] weight', w))
                d.append(('Software', fooocus_version.full_version))
                d.append(('Execution Time', f"{execution_time:.2f} seconds"))
                log(x, d, metadata_string, save_metadata_json, save_metadata_image, keep_input_names, input_image_filename, output_format)

            if not same_seed_for_all:
                seed += 1
            results += imgs

        outputs.append(['results', results])
        outputs.append(['metadatas', metadata_strings])
        return

    while True:
        time.sleep(0.01)
        if len(buffer) > 0:
            task = buffer.pop(0)
            handler(task)
    pass


threading.Thread(target=worker, daemon=True).start()
