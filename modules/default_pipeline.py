import modules.core as core
import os
import gc
import torch
import numpy as np
import modules.path

from comfy.model_base import SDXL, SDXLRefiner
from comfy.model_management import soft_empty_cache
from modules.settings import default_settings
from modules.patch import set_comfy_adm_encoding, set_fooocus_adm_encoding, cfg_patched
from modules.expansion import FooocusExpansion


xl_base: core.StableDiffusionModel = None
xl_base_hash = ''

xl_refiner: core.StableDiffusionModel = None
xl_refiner_hash = ''

xl_base_patched: core.StableDiffusionModel = None
xl_base_patched_hash = ''

clip_vision: core.StableDiffusionModel = None
clip_vision_hash = ''

controlnet_canny: core.StableDiffusionModel = None
controlnet_canny_hash = ''

controlnet_depth: core.StableDiffusionModel = None
controlnet_depth_hash = ''


def refresh_base_model(name):
    global xl_base, xl_base_hash, xl_base_patched, xl_base_patched_hash
    if xl_base_hash == str(name):
        return

    filename = os.path.join(modules.path.modelfile_path, name)

    if xl_base is not None:
        xl_base.to_meta()
        xl_base = None

    xl_base = core.load_model(filename)
    if not isinstance(xl_base.unet.model, SDXL):
        print('Model not supported. Fooocus only support SDXL model as the base model.')
        xl_base = None
        xl_base_hash = ''
        refresh_base_model(modules.path.default_base_model_name)
        xl_base_hash = name
        xl_base_patched = xl_base
        xl_base_patched_hash = ''
        return

    xl_base_hash = name
    xl_base_patched = xl_base
    xl_base_patched_hash = ''
    print(f'Base model loaded: {xl_base_hash}')
    return


def refresh_refiner_model(name):
    global xl_refiner, xl_refiner_hash
    if xl_refiner_hash == str(name):
        return

    if name == 'None':
        xl_refiner = None
        xl_refiner_hash = ''
        print(f'Refiner unloaded.')
        return

    filename = os.path.join(modules.path.modelfile_path, name)

    if xl_refiner is not None:
        xl_refiner.to_meta()
        xl_refiner = None

    xl_refiner = core.load_model(filename)
    if not isinstance(xl_refiner.unet.model, SDXLRefiner):
        print('Model not supported. Fooocus only support SDXL refiner as the refiner.')
        xl_refiner = None
        xl_refiner_hash = ''
        print(f'Refiner unloaded.')
        return

    xl_refiner_hash = name
    print(f'Refiner model loaded: {xl_refiner_hash}')

    xl_refiner.vae.first_stage_model.to('meta')
    xl_refiner.vae = None
    return


def refresh_loras(loras):
    global xl_base, xl_base_patched, xl_base_patched_hash
    if xl_base_patched_hash == str(loras):
        return

    model = xl_base
    for name, weight in loras:
        if name == 'None':
            continue

        filename = os.path.join(modules.path.lorafile_path, name)
        model = core.load_lora(model, filename, strength_model=weight, strength_clip=weight)
    xl_base_patched = model
    xl_base_patched_hash = str(loras)
    print(f'LoRAs loaded: {xl_base_patched_hash}')

    return


def refresh_clip_vision():
    global clip_vision, clip_vision_hash
    if clip_vision_hash == str(clip_vision):
        return

    model_name = modules.path.default_clip_vision_name
    filename = os.path.join(modules.path.clip_vision_path, model_name)
    clip_vision = core.load_clip_vision(filename)

    clip_vision_hash = model_name
    print(f'CLIP Vision model loaded: {clip_vision_hash}')

    return


def refresh_controlnet_canny(name=None):
    global controlnet_canny, controlnet_canny_hash
    if controlnet_canny_hash == str(controlnet_canny):
        return

    model_name = modules.path.default_controlnet_canny_name if name == None else name
    filename = os.path.join(modules.path.controlnet_path, model_name)
    controlnet_canny = core.load_controlnet(filename)

    controlnet_canny_hash = model_name
    print(f'ControlNet model loaded: {controlnet_canny_hash}')

    return


def refresh_controlnet_depth(name=None):
    global controlnet_depth, controlnet_depth_hash
    if controlnet_depth_hash == str(controlnet_depth):
        return

    model_name = modules.path.default_controlnet_depth_name if name == None else name
    filename = os.path.join(modules.path.controlnet_path, model_name)
    controlnet_depth = core.load_controlnet(filename)

    controlnet_depth_hash = model_name
    print(f'ControlNet model loaded: {controlnet_depth_hash}')

    return


refresh_base_model(default_settings['base_model'])

expansion_model = FooocusExpansion()


def expand_txt(*args, **kwargs):
    return expansion_model(*args, **kwargs)


def process_prompt(text, base_clip_skip, refiner_clip_skip, prompt_strength=1.0, revision=False, revision_strengths=[], clip_vision_outputs=[]):
    xl_base_patched.clip.clip_layer(base_clip_skip)
    base_cond = core.encode_prompt_condition(clip=xl_base_patched.clip, prompt=text)
    if prompt_strength >= 0 and prompt_strength < 1.0:
        base_cond = core.set_conditioning_strength(base_cond, prompt_strength)

    if revision:
        set_comfy_adm_encoding()
        for i in range(len(clip_vision_outputs)):
            if revision_strengths[i % 4] != 0:
                base_cond = core.apply_adm(base_cond, clip_vision_outputs[i % 4], revision_strengths[i % 4], 0)
    else:
        set_fooocus_adm_encoding()

    if xl_refiner is not None:
        xl_refiner.clip.clip_layer(refiner_clip_skip)
        refiner_cond = core.encode_prompt_condition(clip=xl_refiner.clip, prompt=text)
        if prompt_strength >= 0 and prompt_strength < 1.0:
            refiner_cond = core.set_conditioning_strength(refiner_cond, prompt_strength)
    else:
        refiner_cond = None
    return base_cond, refiner_cond


@torch.no_grad()
def process_diffusion(positive_cond, negative_cond, steps, switch, width, height, image_seed, sampler_name, scheduler, cfg,
    img2img, input_image, start_step, denoise, revision, clip_vision_outputs, revision_strengths,
    control_lora_canny, canny_edge_low, canny_edge_high, canny_start, canny_stop, canny_strength,
    control_lora_depth, depth_start, depth_stop, depth_strength, callback):

    if xl_base is not None:
        xl_base.unet.model_options['sampler_cfg_function'] = cfg_patched

    if xl_base_patched is not None:
        xl_base_patched.unet.model_options['sampler_cfg_function'] = cfg_patched

    if xl_refiner is not None:
        xl_refiner.unet.model_options['sampler_cfg_function'] = cfg_patched

    positive_conditions = positive_cond[0]
    negative_conditions = negative_cond[0]

    if img2img and input_image != None:
        initial_latent = core.encode_vae(vae=xl_base_patched.vae, pixels=input_image)
        force_full_denoise = False
    else:
        initial_latent = core.generate_empty_latent(width=width, height=height, batch_size=1)
        force_full_denoise = True
        denoise = None

    if control_lora_canny and input_image != None:
        edges_image = core.detect_edge(input_image, canny_edge_low, canny_edge_high)
        positive_conditions, negative_conditions = core.apply_controlnet(positive_conditions, negative_conditions,
            controlnet_canny, edges_image, canny_strength, canny_start, canny_stop)

    if control_lora_depth and input_image != None:
        positive_conditions, negative_conditions = core.apply_controlnet(positive_conditions, negative_conditions,
            controlnet_depth, input_image, depth_strength, depth_start, depth_stop)

    if xl_refiner is not None:
        positive_conditions_refiner = positive_cond[1]
        negative_conditions_refiner = negative_cond[1]

        sampled_latent = core.ksampler_with_refiner(
            model=xl_base_patched.unet,
            positive=positive_conditions,
            negative=negative_conditions,
            refiner=xl_refiner.unet,
            refiner_positive=positive_conditions_refiner,
            refiner_negative=negative_conditions_refiner,
            refiner_switch_step=switch,
            latent=initial_latent,
            steps=steps, start_step=start_step, last_step=steps,
            disable_noise=False, force_full_denoise=force_full_denoise, denoise=denoise,
            seed=image_seed,
            sampler_name=sampler_name,
            scheduler=scheduler,
            cfg=cfg,
            callback_function=callback
        )
    else:
        sampled_latent = core.ksampler(
            model=xl_base_patched.unet,
            positive=positive_conditions,
            negative=negative_conditions,
            latent=initial_latent,
            steps=steps, start_step=start_step, last_step=steps,
            disable_noise=False, force_full_denoise=force_full_denoise, denoise=denoise,
            seed=image_seed,
            sampler_name=sampler_name,
            scheduler=scheduler,
            cfg=cfg,
            callback_function=callback
        )

    decoded_latent = core.decode_vae(vae=xl_base_patched.vae, latent_image=sampled_latent)
    images = core.image_to_numpy(decoded_latent)

    gc.collect()
    soft_empty_cache()

    return images
