"""
run_locally has a check that makes it invalid when we do the taskset thing. Annoyingly, best thing I can think to do is
disable the sanity check when tasksetting.

This requires having run `update_ulimit_tmux` first, because it removes quotes. Sadly, if any entry in the command_list
has spaces, it puts it in quotes, because python is annoying.


"""
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--venv_dir', type=str, default="../venv")
parser.add_argument('--python_version', type=str, default="python3.8", help="in `venv/lib`, helps make path.")

args = parser.parse_args()
venv_dir = args.venv_dir
python_version = args.python_version
local_run_dir = os.path.join(venv_dir, f"lib/{python_version}/site-packages/launchpad/launch/run_locally/")

file_name = os.path.join(local_run_dir, "run_locally.py")
cached_filename = os.path.join(local_run_dir, "run_locally.py.old")

old_str = r"""
def run_commands_locally(commands: Sequence[CommandToLaunch], terminal=None):
  # Minimally validate all the commands before executing any of them. This also
  # gives better errors in the case that a terminal implementation executes
  # the commands via a wrapper.
  for command in commands:
    if not os.access(command.command_as_list[0], os.X_OK):
      raise ValueError("Unable to execute '%s'" % command.command_as_list[0])
  return _LOCAL_LAUNCHER_MAP[_get_terminal(terminal)](commands)
"""

new_str = r"""
def run_commands_locally(commands: Sequence[CommandToLaunch], terminal=None):
  # Minimally validate all the commands before executing any of them. This also
  # gives better errors in the case that a terminal implementation executes
  # the commands via a wrapper.
  for command in commands:
    if not os.access(command.command_as_list[0], os.X_OK):
      if command.command_as_list[0].startswith('taskset'):
        print('taskset command detected, skipping validation in site-packages/launchpad/launch/run_locally/run_locally.py')
        continue
      raise ValueError("Unable to execute '%s'" % command.command_as_list[0])
  return _LOCAL_LAUNCHER_MAP[_get_terminal(terminal)](commands)
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
    print('old version written to run_locally.py.old')

for og_txt, new_txt in replace_tuples:
  contents = contents.replace(og_txt, new_txt)
  if og_txt not in contents:
    print(og_txt)


with open(file_name, "w") as f:
  f.write(contents)

print('new version written to run_locally.py')