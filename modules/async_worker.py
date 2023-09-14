import os
import threading
import json
import torch
import numpy as np

import modules.core as core
import modules.constants as constants

from PIL import Image, ImageOps
from modules.resolutions import annotate_resolution_string, string_to_dimensions
from modules.settings import default_settings
from comfy.model_management import InterruptProcessingException, throw_exception_if_processing_interrupted


buffer = []
outputs = []


def get_image(path, megapixels=1.0):
    image = None
    with open(path, 'rb') as image_file:
        pil_image = Image.open(image_file)
        image = ImageOps.exif_transpose(pil_image)
        image_file.close()
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        image = core.upscale(image, megapixels)
    return image


def worker():
    global buffer, outputs

    import time
    import shared
    import random
    import copy
    import modules.default_pipeline as pipeline
    import modules.path
    import modules.patch
    import fooocus_version

    from modules.resolutions import get_resolution_string, resolutions
    from modules.sdxl_styles import apply_style
    from modules.private_logger import log
    from modules.expansion import safe_str
    from modules.util import join_prompts, remove_empty_str

    try:
        async_gradio_app = shared.gradio_root
        flag = f'''App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'''
        if async_gradio_app.share:
            flag += f''' or {async_gradio_app.share_url}'''
        print(flag)
    except Exception as e:
        print(e)


    def progressbar(number, text):
        print(f'[Fooocus] {text}')
        outputs.append(['preview', (number, text, None)])


    @torch.no_grad()
    def handler(task):
        prompt, negative_prompt, style_selections, performance, resolution, image_number, image_seed, \
        sharpness, sampler_name, scheduler, custom_steps, custom_switch, cfg, \
        base_model_name, refiner_model_name, base_clip_skip, refiner_clip_skip, \
        l1, w1, l2, w2, l3, w3, l4, w4, l5, w5, save_metadata_json, save_metadata_image, \
        img2img_mode, img2img_start_step, img2img_denoise, \
        revision_mode, positive_prompt_strength, negative_prompt_strength, revision_strength_1, revision_strength_2, \
        revision_strength_3, revision_strength_4, same_seed_for_all, output_format, \
        control_lora_canny, canny_edge_low, canny_edge_high, canny_start, canny_stop, canny_strength, canny_model, \
        control_lora_depth, depth_start, depth_stop, depth_strength, depth_model, use_expansion, \
        input_gallery, revision_gallery, keep_input_names = task

        loras = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]

        raw_style_selections = copy.deepcopy(style_selections)

        use_style = len(style_selections) > 0

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


        progressbar(1, 'Initializing ...')

        raw_prompt = prompt
        raw_negative_prompt = negative_prompt

        prompts = remove_empty_str([safe_str(p) for p in prompt.split('\n')], default='')
        negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.split('\n')], default='')

        prompt = prompts[0]
        negative_prompt = negative_prompts[0]

        extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
        extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

        try:
            seed = int(image_seed) 
        except Exception as e:
            seed = -1
        if not isinstance(seed, int) or seed < constants.MIN_SEED or seed > constants.MAX_SEED:
            seed = random.randint(constants.MIN_SEED, constants.MAX_SEED)


        progressbar(3, 'Loading models ...')
        pipeline.refresh_base_model(base_model_name)
        pipeline.refresh_refiner_model(refiner_model_name)
        pipeline.refresh_loras(loras)
        pipeline.set_clip_skips(base_clip_skip, refiner_clip_skip)
        if revision_mode:
            pipeline.refresh_clip_vision()
        if control_lora_canny:
            pipeline.refresh_controlnet_canny(canny_model)
        if control_lora_depth:
            pipeline.refresh_controlnet_depth(depth_model)


        clip_vision_outputs = []
        if revision_mode:
            revision_images_paths = list(map(lambda x: x['name'], revision_gallery))
            revision_images_filenames = list(map(lambda path: os.path.basename(path), revision_images_paths))
            revision_strengths = [revision_strength_1, revision_strength_2, revision_strength_3, revision_strength_4]
            for i in range(revision_gallery_size):
                progressbar(4, f'Revision for image {i + 1} ...')
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


        pipeline.clear_all_caches()

        progressbar(5, 'Processing prompts ...')

        positive_basic_workloads = []
        negative_basic_workloads = []

        if use_style:
            for s in style_selections:
                p, n = apply_style(s, positive=prompt)
                positive_basic_workloads.append(p)
                negative_basic_workloads.append(n)
        else:
            positive_basic_workloads.append(prompt)

        negative_basic_workloads.append(negative_prompt)  # Always use independent workload for negative.

        positive_basic_workloads = positive_basic_workloads + extra_positive_prompts
        negative_basic_workloads = negative_basic_workloads + extra_negative_prompts

        positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=prompt)
        negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=negative_prompt)

        positive_top_k = len(positive_basic_workloads)
        negative_top_k = len(negative_basic_workloads)

        tasks = [dict(
            task_seed=seed if same_seed_for_all else seed + i,
            positive=positive_basic_workloads,
            negative=negative_basic_workloads,
            expansion='',
            c=[None, None],
            uc=[None, None],
        ) for i in range(image_number)]

        if use_expansion:
            for i, t in enumerate(tasks):
                progressbar(5, f'Preparing Fooocus text #{i + 1} ...')
                expansion = pipeline.expansion(prompt, t['task_seed'])
                print(f'[Prompt Expansion] New suffix: {expansion}')
                t['expansion'] = expansion
                t['positive'] = copy.deepcopy(t['positive']) + [join_prompts(prompt, expansion)]  # Deep copy.

        for i, t in enumerate(tasks):
            progressbar(7, f'Encoding base positive #{i + 1} ...')
            t['c'][0] = pipeline.clip_encode(sd=pipeline.xl_base_patched, texts=t['positive'],
                                             pool_top_k=positive_top_k)

        for i, t in enumerate(tasks):
            progressbar(9, f'Encoding base negative #{i + 1} ...')
            t['uc'][0] = pipeline.clip_encode(sd=pipeline.xl_base_patched, texts=t['negative'],
                                              pool_top_k=negative_top_k)

        if pipeline.xl_refiner is not None:
            for i, t in enumerate(tasks):
                progressbar(11, f'Encoding refiner positive #{i + 1} ...')
                t['c'][1] = pipeline.clip_encode(sd=pipeline.xl_refiner, texts=t['positive'],
                                                 pool_top_k=positive_top_k)

            for i, t in enumerate(tasks):
                progressbar(13, f'Encoding refiner negative #{i + 1} ...')
                t['uc'][1] = pipeline.clip_encode(sd=pipeline.xl_refiner, texts=t['negative'],
                                                  pool_top_k=negative_top_k)

        for i, t in enumerate(tasks):
            progressbar(13, f'Applying prompt strengths #{i + 1} ...')
            t['c'][0], t['c'][1] = pipeline.apply_prompt_strength(t['c'][0], t['c'][1], positive_prompt_strength)
            t['uc'][0], t['uc'][1] = pipeline.apply_prompt_strength(t['uc'][0], t['uc'][1], negative_prompt_strength)

        for i, t in enumerate(tasks):
            progressbar(13, f'Applying Revision #{i + 1} ...')
            t['c'][0] = pipeline.apply_revision(t['c'][0], revision_mode, revision_strengths, clip_vision_outputs)


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
            try:
                resolution = annotate_resolution_string(resolution)
            except Exception as e:
                print(f'Problem with resolution definition: "{resolution}", reverting to default: ' + default_settings['resolution'])
                resolution = default_settings['resolution']
        width, height = string_to_dimensions(resolution)

        img2img_megapixels = width * height / 2**20
        if img2img_megapixels < constants.MIN_MEGAPIXELS:
            img2img_megapixels = constants.MIN_MEGAPIXELS
        elif img2img_megapixels > constants.MAX_MEGAPIXELS:
            img2img_megapixels = constants.MAX_MEGAPIXELS

        pipeline.clear_all_caches()  # save memory

        results = []
        metadata_strings = []
        all_steps = steps * image_number

        def callback(step, x0, x, total_steps, y):
            throw_exception_if_processing_interrupted()
            done_steps = current_task_idx * steps + step
            outputs.append(['preview', (
                int(15.0 + 85.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {current_task_idx + 1}-th Sampling',
                y)])

        outputs.append(['preview', (13, 'Starting tasks ...', None)])
        stop_batch = False
        for current_task_idx, task in enumerate(tasks):
            if img2img_mode or control_lora_canny or control_lora_depth:
                input_gallery_entry = input_gallery[current_task_idx % input_gallery_size]
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
                input_image = get_image(input_image_path, img2img_megapixels)

            execution_start_time = time.perf_counter()
            try:
                imgs = pipeline.process_diffusion(task['c'], task['uc'], steps, switch, width, height, task['task_seed'],
                    sampler_name, scheduler, cfg, img2img_mode, input_image, start_step, denoise,
                    control_lora_canny, canny_edge_low, canny_edge_high, canny_start, canny_stop, canny_strength,
                    control_lora_depth, depth_start, depth_stop, depth_strength, callback=callback)
            except InterruptProcessingException as iex:
                print('Processing interrupted')
                stop_batch = True
                imgs = []
            execution_time = time.perf_counter() - execution_start_time
            print(f'Prompt executed in {execution_time:.2f} seconds')

            metadata = {
                'prompt': raw_prompt, 'negative_prompt': raw_negative_prompt, 'styles': raw_style_selections,
                'seed': task['task_seed'], 'width': width, 'height': height,
                'sampler': sampler_name, 'scheduler': scheduler, 'performance': performance,
                'steps': steps, 'switch': switch, 'sharpness': sharpness, 'cfg': cfg,
                'base_clip_skip': base_clip_skip, 'refiner_clip_skip': refiner_clip_skip,
                'base_model': base_model_name, 'refiner_model': refiner_model_name,
                'l1': l1, 'w1': w1, 'l2': l2, 'w2': w2, 'l3': l3, 'w3': w3,
                'l4': l4, 'w4': w4, 'l5': l5, 'w5': w5, 'img2img': img2img_mode, 'revision': revision_mode,
                'positive_prompt_strength': positive_prompt_strength, 'negative_prompt_strength': negative_prompt_strength,
                'control_lora_canny': control_lora_canny, 'control_lora_depth': control_lora_depth,
                'prompt_expansion': use_expansion
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
                    ('Prompt', raw_prompt),
                    ('Negative Prompt', raw_negative_prompt),
                    ('Fooocus V2 (Prompt Expansion)', task['expansion']),
                    ('Styles', str(raw_style_selections)),
                    ('Seed', task['task_seed']),
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
                    ('Prompt Strengths', (positive_prompt_strength, negative_prompt_strength)),
                    ('Canny', (control_lora_canny, canny_edge_low, canny_edge_high, canny_start, canny_stop,
                        canny_strength, canny_model, input_image_filename) if control_lora_canny else (control_lora_canny)),
                    ('Depth', (control_lora_depth, depth_start, depth_stop, depth_strength, depth_model, input_image_filename) if control_lora_depth else (control_lora_depth))
                ]
                for n, w in loras:
                    if n != 'None':
                        d.append((f'LoRA [{n}] weight', w))
                d.append(('Software', fooocus_version.full_version))
                d.append(('Execution Time', f'{execution_time:.2f} seconds'))
                log(x, d, 3, metadata_string, save_metadata_json, save_metadata_image, keep_input_names, input_image_filename, output_format)

            results += imgs

            if stop_batch:
                break

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
