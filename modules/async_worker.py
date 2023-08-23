import os
import threading
import json

import modules.core as core
import modules.constants as constants


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
        prompt, negative_prompt, style, performance, \
        resolution, image_number, image_seed, sharpness, sampler_name, scheduler, \
        custom_steps, custom_switch, cfg, \
        base_model_name, refiner_model_name, base_clip_skip, refiner_clip_skip, \
        l1, w1, l2, w2, l3, w3, l4, w4, l5, w5, save_metadata_json, save_metadata_png, \
        img2img_mode, img2img_start_step, img2img_denoise, gallery = task

        loras = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]

        modules.patch.sharpness = sharpness

        pipeline.refresh_base_model(base_model_name)
        pipeline.refresh_refiner_model(refiner_model_name)
        pipeline.refresh_loras(loras)
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

        gallery_size = len(gallery)
        for i in range(image_number):
            if img2img_mode and gallery_size > 0:
                start_step = round(steps * img2img_start_step)
                denoise = img2img_denoise
                gallery_entry = gallery[i % gallery_size]
                input_image_path = gallery_entry['name']
            else:
                start_step = 0
                denoise = None
                input_image_path = None

            imgs = pipeline.process(p_txt, n_txt, steps, switch, width, height, seed, sampler_name, scheduler,
                cfg, base_clip_skip, refiner_clip_skip, input_image_path, start_step, denoise, callback=callback)

            metadata = {
                'prompt': prompt, 'negative_prompt': negative_prompt, 'style': style,
                'seed': seed, 'width': width, 'height': height, 'p_txt': p_txt, 'n_txt': n_txt,
                'sampler': sampler_name, 'scheduler': scheduler, 'performance': performance,
                'steps': steps, 'switch': switch, 'sharpness': sharpness, 'cfg': cfg,
                'base_clip_skip': base_clip_skip, 'refiner_clip_skip': refiner_clip_skip,
                'base_model': base_model_name, 'refiner_model': refiner_model_name,
                'l1': l1, 'w1': w1, 'l2': l2, 'w2': w2, 'l3': l3, 'w3': w3,
                'l4': l4, 'w4': w4, 'l5': l5, 'w5': w5, 'img2img': img2img_mode,
                'start_step': start_step, 'denoise': denoise, 'input_image': None if input_image_path == None else os.path.basename(input_image_path),
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
                    ('Image-2-Image', (img2img_mode, start_step, denoise, metadata['input_image']))
                ]
                for n, w in loras:
                    if n != 'None':
                        d.append((f'LoRA [{n}] weight', w))
                d.append(('Software', fooocus_version.full_version))
                log(x, d, metadata_string, save_metadata_json, save_metadata_png)

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
