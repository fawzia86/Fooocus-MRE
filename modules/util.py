import random
import os
import modules.path

from datetime import datetime, timedelta


def remove_empty_str(items, default=None):
    items = [x for x in items if x != ""]
    if len(items) == 0 and default is not None:
        return [default]
    return items


def join_prompts(*args, **kwargs):
    prompts = [str(x) for x in args if str(x) != ""]
    if len(prompts) == 0:
        return ""
    if len(prompts) == 1:
        return prompts[0]
    return ', '.join(prompts)


def generate_temp_filename(folder='./outputs/', extension='png', base=None):
    current_time = datetime.now()
    date_string = current_time.strftime("%Y-%m-%d")
    if base == None:
        time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        random_number = random.randint(1000, 9999)
        filename = f"{time_string}_{random_number}.{extension}"
    else:
        filename = f"{os.path.splitext(base)[0]}.{extension}"
    result = os.path.join(folder, date_string, filename)
    return date_string, os.path.abspath(os.path.realpath(result)), filename


def get_log_path(time):
    current_dir = os.path.abspath(os.curdir)
    outputs_dir = os.path.abspath(modules.path.temp_outputs_path)
    if outputs_dir.startswith(current_dir):
        folder = os.path.relpath(outputs_dir, current_dir)
    else:
        folder = outputs_dir
    return os.path.join(folder, time.strftime("%Y-%m-%d"), 'log.html')


def get_current_log_path():
    time = datetime.now()
    return get_log_path(time)


def get_previous_log_path():
    time = datetime.now() - timedelta(days=1)
    return get_log_path(time)
