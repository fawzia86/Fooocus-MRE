import os
import modules.path
import json

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from modules.util import generate_temp_filename


def log(img, dic, save_metadata, metadata):
    date_string, local_temp_filename, only_name = generate_temp_filename(folder=modules.path.temp_outputs_path, extension='png')
    os.makedirs(os.path.dirname(local_temp_filename), exist_ok=True)

    if save_metadata == 'JSON':
        json_path = local_temp_filename.replace('.png', '.json')
        with open(json_path, 'w') as json_file:
            json.dump(metadata, json_file)
            json_file.close()
    
    if save_metadata == 'PNG':
        pnginfo = PngInfo()
        pnginfo.add_text("Comment", json.dumps(metadata))
    else:
        pnginfo = None

    Image.fromarray(img).save(local_temp_filename, pnginfo=pnginfo)

    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log.html')

    if not os.path.exists(html_name):
        with open(html_name, 'a+') as f:
            f.write(f"<p>Fooocus Log {date_string} (private)</p>\n")
            f.write(f"<p>All images do not contain any hidden data.</p>")

    with open(html_name, 'a+') as f:
        div_name = only_name.replace('.', '_')
        f.write(f'<div id="{div_name}"><hr>\n')
        f.write(f"<p>{only_name}</p>\n")
        i = 0
        for k, v in dic:
            if i < 2:
                f.write(f"<p>{k}: <b>{v}</b> </p>\n")
            else:
                if i % 2 == 0:
                    f.write(f"<p>{k}: <b>{v}</b>, ")
                else:
                    f.write(f"{k}: <b>{v}</b></p>\n")
            i += 1
        f.write(f"<p><img src=\"{only_name}\" width=512 onerror=\"document.getElementById('{div_name}').style.display = 'none';\"></img></p></div>\n")

    print(f'Image generated with private log at: {html_name}')

    return
