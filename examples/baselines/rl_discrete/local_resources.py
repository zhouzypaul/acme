from launchpad.nodes.python.local_multi_processing import PythonProcess
from absl import flags
import psutil

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'limit_all_cpus', False,
    'Separate from one_cpu_per_actor, this puts limits on the other nodes. Arbitrarily chooses 4 CPUs per node.')

# Let's just write out what we need. We'll start with just actors.
# num_actors, num_actors_per_node, then we calculate num_actor_nodes.
# Then we need the CPU range for actors.
# Then we have a helper function that divides it up correctly
# Then we change the interpreter. Nice.
# We also want learner_gpus and 

def pin_process_to_cpu(python_process, cpu_num):
  interpreter = python_process.absolute_interpreter_path
  print('old interpreter', interpreter)
  new_interpreter_path = f"taskset -c {cpu_num} {interpreter}"
  # python_process._absolute_interpreter_path = interpreter
  python_process._absolute_interpreter_path = new_interpreter_path
  print('new interpreter', new_interpreter_path)
  return python_process


def make_process_dict(gpu_str="-1", pin_to=None):
  # Has ability to modify process to pin to specified CPUs using taskset. Good for controlling jax thread numbers
  # print('\n\n\nDONT FORGET ABOUT THIS YOU JERKS\n\n\n')
  process = PythonProcess(env={
    "CUDA_VISIBLE_DEVICES": gpu_str,
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    "ACME_ID": FLAGS.acme_id,
    "ACME_DIR": FLAGS.acme_dir,
  # "JAX_DISABLE_JIT": "1",
  },
  )
  if pin_to is not None:
    if isinstance(pin_to, int):
      pin_to = [pin_to]
    print('cpu_count: ', psutil.cpu_count())
    p = psutil.Process()
    all_cpu_ids = p.cpu_affinity()
    these_cpu_ids = [str(all_cpu_ids[i % len(all_cpu_ids)]) for i in pin_to]
    these_cpu_ids = list(set(these_cpu_ids))
    cpu_str = ",".join(these_cpu_ids)
    process = pin_process_to_cpu(process, cpu_str)

  return process

    

def make_actor_resources(num_actors, one_cpu_per_actor=False):
  # If you don't specify CPU range, then just do them all like before I guess.
  # What if I always modify it? I think that's better actually.
  actor_gpu_ids = FLAGS.actor_gpu_ids
  assert isinstance(actor_gpu_ids, list) and len(actor_gpu_ids) > 0, actor_gpu_ids

  if one_cpu_per_actor:
    import psutil
    print('cpu_count: ', psutil.cpu_count())
    p = psutil.Process()
    cpu_ids = p.cpu_affinity()
    print('cpu_ids: ', cpu_ids)

  process_dict = {}
  for actor_num in range(num_actors):
    print('doing the other way!')
    gpu_id = actor_gpu_ids[actor_num % len(actor_gpu_ids)]
    actor_key = f"actor_{actor_num}"
    process = make_process_dict(gpu_id)
    if one_cpu_per_actor:
      cpu_id = cpu_ids[actor_num % len(cpu_ids)]
      process = pin_process_to_cpu(process, cpu_id)

    process_dict[actor_key] = process

  return process_dict


def get_local_resources(launch_type):
  num_actors = FLAGS.num_actors
  num_actors_per_node = FLAGS.num_actors_per_node
  one_cpu_per_actor = FLAGS.one_cpu_per_actor

  assert num_actors_per_node == 1, num_actors_per_node

  assert launch_type in ('local_mp', 'local_mt'), launch_type
  from launchpad.nodes.python.local_multi_processing import PythonProcess
  if launch_type == 'local_mp':
    if FLAGS.limit_all_cpus:
      cpu_dict = {
        'learner'   : [0, 1, 2, 3],
        'replay'    : [4, 5, 6, 7],
        'counter' : [8, 9],
        'evaluator': [10],
        'cfn': [11, 12, 13, 14],
      }
    else:
      cpu_dict = {}
    local_resources = {
      "learner": make_process_dict(",".join(FLAGS.learner_gpu_ids), pin_to=cpu_dict.get('learner')),
      "counter": make_process_dict(pin_to=cpu_dict.get('counter')),
      "replay": make_process_dict(pin_to=cpu_dict.get('replay')),
      "evaluator": make_process_dict(pin_to=cpu_dict.get('evaluator')),
      "cfn": make_process_dict(FLAGS.cfn_gpu_id, pin_to=cpu_dict.get('cfn'))
    }
    # TODO: Be able to choose actor GPU so that we can compare 1 and 2 GPU utilization etc.
    actor_resources = make_actor_resources(num_actors=num_actors, one_cpu_per_actor=one_cpu_per_actor)
    local_resources.update(actor_resources)
  else:
    local_resources = {}

  print('local_resources keys: ', local_resources.keys())
  return local_resources

if __name__ == '__main__':
  pass
