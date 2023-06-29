"""
As from https://github.com/deepmind/acme/issues/279
"""

import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--venv_dir', type=str, default="../venv")
parser.add_argument('--python_version', type=str, default="python3.8", help="in `venv/lib`, helps make path.")

args = parser.parse_args()
venv_dir = args.venv_dir
python_version = args.python_version
courier_utils_dir = os.path.join(venv_dir, f"lib/{python_version}/site-packages/launchpad/nodes/courier")

file_name = os.path.join(courier_utils_dir, "courier_utils.py")
cached_filename = os.path.join(courier_utils_dir, "courier_utils.py.old")

replace_tuples = [("timeout.microseconds/1000, pad_batch)",
                   "timeout.microseconds//1000, pad_batch)"),
                  ("timeout, pad_batch)", "timeout.microseconds // 1000, pad_batch)")]
                #   ("timeout, pad_batch)", "timeout.microseconds // 100, pad_batch)")] # Kinalmehta had the last line without the divide...

if not os.path.exists(file_name):
    print('file not found. Maybe not installed, maybe wrong version of python in dir string.')
if os.path.exists(cached_filename):
    raise Exception("Seems like you probably already did the replacement already? old file already there")


with open(file_name) as f:
  contents = f.read()

with open(cached_filename, "w") as f:
    f.write(contents)
    print('old version written to courier_utils.py.old')

for og_txt, new_txt in replace_tuples:
  contents = contents.replace(og_txt, new_txt)
  if og_txt not in contents:
    print(og_txt)


with open(file_name, "w") as f:
  f.write(contents)

print('new version written to courier_utils.py')