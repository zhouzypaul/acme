raise Exception("This script works. But, ccv sysadmin were nice enough to raise the ulimit "
                "for everyone in the gdk group, so should no longer be necessary so long as you "
                "use the right options")

import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--venv_dir', type=str, default="../venv")
parser.add_argument('--python_version', type=str, default="python3.8", help="in `venv/lib`, helps make path.")

args = parser.parse_args()
venv_dir = args.venv_dir
python_version = args.python_version
local_run_dir = os.path.join(venv_dir, f"lib/{python_version}/site-packages/launchpad/launch/run_locally/")

file_name = os.path.join(local_run_dir, "launch_local_tmux.py")
cached_filename = os.path.join(local_run_dir, "launch_local_tmux.py.old")

old_str = r"""
    inner_command = f'{command_str}; echo "{command_str}"; exec $SHELL'

    window_name = command_to_launch.title
"""

new_str = r"""
    print('[launch_local_tmux] doing something real bad -- removing quotes')
    command_str = command_str.replace('"', '')
    
    inner_command = f'{command_str}; echo "{command_str}"; exec $SHELL'
    print('[launch_local_tmux] inner_command: ', inner_command)

    print('[launch_local_tmux] changing ulimit here always')
    inner_command = f'ulimit -u 100000 ; {inner_command}'
    print('[launch_local_tmux] inner_command: ', inner_command)
    window_name = command_to_launch.title
"""

replace_tuples = [
  (old_str, new_str),
]

if not os.path.exists(file_name):
    print('file not found. Maybe not installed, maybe wrong version of python in dir string.')
if os.path.exists(cached_filename):
    raise Exception("Seems like you probably already did the replacement already? old file already there")


with open(file_name) as f:
  contents = f.read()

with open(cached_filename, "w") as f:
    f.write(contents)
    print('old version written to launch_local_tmux.py.old')

for og_txt, new_txt in replace_tuples:
  contents = contents.replace(og_txt, new_txt)
  if og_txt not in contents:
    print(og_txt)


with open(file_name, "w") as f:
  f.write(contents)

print('new version written to launch_local_tmux.py')