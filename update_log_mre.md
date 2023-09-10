### 1.0.50 MRE
* Renamed Raw Mode (enabled by default in vanilla) to Prompt Expansion (disabled by default in MRE)

### 1.0.45.1 MRE

* Added support for reading styles from JSON files.
* Added support for playing audio when generation is finished (ported from SD web UI).
* Fixed joining negative prompts.
* Increased Control-LoRAs strength range.
* Allowed passing parameters to Comfy.
* Added links in Misc tab
* Updated Comfy.

### 1.0.45 MRE

* Added support for custom resolutions and custom resolutions list.
* Updated Comfy.
* Added MRE changelog.

### 1.0.43 MRE

* Added support for Control-LoRA: Depth.
* Added Canny and Depth model selection.
* Added ability to stop image generation.
* Added support for generate forever mode (ported from SD web UI).
* Added information about prompt execution time.
* Changed default Control-LoRA rank to 128 (allows using both on 8 GB VRAM GPUs).
* Fixed problems with random seed precision.
* Increased maximum number of generated images to 128.
* Moved Revision in front of Control-LoRAs.
* Updated Comfy.
* Updated Gradio.
* Updated Colab notebook.

### 1.0.41 MRE

* Added support for Revision (thx to comfyanonymous for helping figuring out issues related to lllyasviel changes).
* Added support for Control-LoRA: Canny.
* Added support for Embeddings.
* Added support for JPEGs (with metadata saving & loading).
* Added support for generating multiple images using same seed (useful in Image-2-Image mode).
* Added support for keeping input files names.
* Added information about aspect ratio to resolution list.
* Added dpmpp_2m sampler.
* Automatically import images into Image-2-Image and Revision modes (if available).
* Automatically set batch size to number of loaded input images.
* Loading only base model on start.
* Updated Comfy.
* Updated Colab notebook.

### 1.0.40 MRE

* Added support for Image-2-Image mode.
* Added support for changing paths (via paths.json).
* Added Custom performance mode.
* Fixed prompt loading.
* Implemented cycling over input images in Image-2-Image mode.
* Made resolution and style selection more compact (thx to runew0lf for hints).
* Simplified initial view.
* Updated Comfy.
* Improvemed VRAM management.
* Corrected Colab notebook.

### 1.0.39 MRE

* Updated Comfy (+DISABLE_SMART_MEMORY).

### 1.0.36 MRE

* Fixed problem with disabled refiner (by youcheng).
* Layout corrections.
* Updated documentation.
* Renamed fork to "Fooocus-MRE".

### 1.0.35 MRE

* Added support for viewing metadata in the UI.
* Added support for loading basic UI settings from file (settings.json).
* Allowed saving metadata in both JSON and PNG file.
* Fixed problem with handling text files encoding (for non-ASCII prompts).

### 1.0.32 MRE

* Added support for loading prompt parameters from JSON and PNG files.
* Added support for changing scheduler (karras, exponential, simple, ddim_uniform).
* Enhanced list of available samplers (added dpmpp_sde_gpu, dpmpp_sde, dpmpp_2s_ancestral, euler, euler_ancestral, heun, dpm_2, dpm_2_ancestral).
* Updated Comfy.

### 1.0.31 MRE

* Added support for dpmpp_3m_sde_gpu, dpmpp_3m_sde, and dpmpp_2m_sde samplers.
* Added support for saving prompt information in PNG metadata or JSON file.
* Added support for changing steps, switch step, and CFG.
* Added support for changing CLIP Skip values.
* Changed resolution list to official SDXL resolutions.
* Enhanced RNG seed range.
