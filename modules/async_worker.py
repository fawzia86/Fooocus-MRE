import os
import threading
import json

import modules.core as core
import modules.constants as constants

from modules.resolutions import annotate_resolution_string


buffer = []
outputs = []


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
        l1, w1, l2, w2, l3, w3, l4, w4, l5, w5, save_metadata_json, save_metadata_png, \
        img2img_mode, img2img_start_step, img2img_denoise, \
        revision_mode, zero_out_positive, zero_out_negative, revision_strength_1, revision_strength_2, \
        revision_strength_3, revision_strength_4, same_seed_for_all, \
        input_gallery, revision_gallery = task

        loras = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]

        modules.patch.sharpness = sharpness

        input_gallery_size = len(input_gallery)
        if input_gallery_size == 0:
            img2img_mode = False
            input_image_path = None

        revision_gallery_size = len(revision_gallery)
        if revision_gallery_size == 0:
            revision_mode = False

        pipeline.refresh_base_model(base_model_name)
        pipeline.refresh_refiner_model(refiner_model_name)
        pipeline.refresh_loras(loras)
        if revision_mode:
            pipeline.refresh_clip_vision()
        pipeline.clean_prompt_cond_caches()

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

        seed = image_seed
        max_seed = 2**63 - 1
        if not isinstance(seed, int):
            seed = random.randint(0, max_seed)
        if seed < 0:
            seed = - seed
        seed = seed % max_seed

        all_steps = steps * image_number

        def callback(step, x0, x, total_steps, y):
            done_steps = i * steps + step
            outputs.append(['preview', (
                int(100.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {i}-th Sampling',
                y)])

        for i in range(image_number):
            if img2img_mode and input_gallery_size > 0:
                input_gallery_entry = input_gallery[i % input_gallery_size]
                input_image_path = input_gallery_entry['name']
                input_image_filename = None if input_image_path == None else os.path.basename(input_image_path)
            else:
                input_image_path = None
                input_image_filename = None

            if revision_mode:
                revision_images_paths = list(map(lambda x: x['name'], revision_gallery))
                revision_images_filenames = list(map(lambda path: os.path.basename(path), revision_images_paths))
                revision_strengths = [revision_strength_1, revision_strength_2, revision_strength_3, revision_strength_4]
            else:
                revision_images_paths = []
                revision_images_filenames = []
                revision_strengths = []

            if img2img_mode:
                start_step = round(steps * img2img_start_step)
                denoise = img2img_denoise
            else:
                start_step = 0
                denoise = None

            imgs = pipeline.process(p_txt, n_txt, steps, switch, width, height, seed, sampler_name, scheduler,
                cfg, base_clip_skip, refiner_clip_skip, img2img_mode, input_image_path, start_step, denoise,
                revision_mode, revision_images_paths, zero_out_positive, zero_out_negative, revision_strengths, callback=callback)

            metadata = {
                'prompt': prompt, 'negative_prompt': negative_prompt, 'style': style,
                'seed': seed, 'width': width, 'height': height, 'p_txt': p_txt, 'n_txt': n_txt,
                'sampler': sampler_name, 'scheduler': scheduler, 'performance': performance,
                'steps': steps, 'switch': switch, 'sharpness': sharpness, 'cfg': cfg,
                'base_clip_skip': base_clip_skip, 'refiner_clip_skip': refiner_clip_skip,
                'base_model': base_model_name, 'refiner_model': refiner_model_name,
                'l1': l1, 'w1': w1, 'l2': l2, 'w2': w2, 'l3': l3, 'w3': w3,
                'l4': l4, 'w4': w4, 'l5': l5, 'w5': w5, 'img2img': img2img_mode,
                'start_step': start_step, 'denoise': denoise, 'input_image': input_image_filename,
                'revision': revision_mode, 'zero_out_positive': zero_out_positive, 'zero_out_negative': zero_out_negative,
                'revision_strength_1': revision_strength_1, 'revision_strength_2': revision_strength_2,
                'revision_strength_3': revision_strength_3, 'revision_strength_4': revision_strength_4,
                'revision_images': revision_images_filenames,
                'software': fooocus_version.full_version
            }
            metadata_string = json.dumps(metadata, ensure_ascii=False)
            metadata_strings.append(metadata_string)

            for x in imgs:
                d = [
                    ('Prompt', prompt),
                    ('Negative Prompt', negative_prompt),
                    ('Style', style),
                    ('Seed', seed),
                    ('Resolution', get_resolution_string(width, height)),
                    ('Performance', str((performance, steps, switch))),
                    ('Sampler & Scheduler', str((sampler_name, scheduler))),
                    ('Sharpness', sharpness),
                    ('CFG & CLIP Skips', str((cfg, base_clip_skip, refiner_clip_skip))),
                    ('Base Model', base_model_name),
                    ('Refiner Model', refiner_model_name),
                    ('Image-2-Image', (img2img_mode, start_step, denoise, input_image_filename)),
                    ('Revision', (revision_mode, zero_out_positive, zero_out_negative, revision_strength_1, revision_strength_2,
                        revision_strength_3, revision_strength_4, revision_images_filenames))
                ]
                for n, w in loras:
                    if n != 'None':
                        d.append((f'LoRA [{n}] weight', w))
                d.append(('Software', fooocus_version.full_version))
                log(x, d, metadata_string, save_metadata_json, save_metadata_png)

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
