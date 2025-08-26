import os
import math
import random
from robocasa.utils.dataset_registry import get_ds_path
from robocasa.utils.dataset_registry import (
    SINGLE_STAGE_TASK_DATASETS,
    MULTI_STAGE_TASK_DATASETS,
)

dataset_cands = []
for task in list(SINGLE_STAGE_TASK_DATASETS.keys()) + list(
    MULTI_STAGE_TASK_DATASETS.keys()
):
    for split in ["train", "test"]:
        for ds_type in ["human_raw", "mg_raw"]:
            ds_path = get_ds_path(task=task, split=split, ds_type=ds_type)
            if ds_path is not None:
                dataset_cands.append(ds_path)

dataset_cands = sorted(dataset_cands)

datasets = []
for ds in dataset_cands:
    exists = False

    groot_dir = os.path.join(os.path.dirname(ds), "groot")
    if (
        os.path.exists(os.path.join(groot_dir, "videos"))
        and os.path.exists(os.path.join(groot_dir, "data"))
        and os.path.exists(os.path.join(groot_dir, "meta"))
    ):
        exists = True

    # groot_raw_dir = os.path.join(os.path.dirname(ds), "groot_raw")
    # if os.path.exists(groot_raw_dir):
    #     exists = True

    if not exists:
        print('"' + groot_dir + '",')
        datasets.append(ds)

print()
print()
print()

command_list = []
for ds in datasets:
    directory = os.path.dirname(ds)
    command = f"python /mnt/amlfs-01/shared/robocasa_benchmark/misc/export_groot/stage1.py --num_procs 25 --dataset {ds}; python /mnt/amlfs-01/shared/robocasa_benchmark/misc/export_groot/stage2.py --directory {directory}"
    command_list.append(command)
    # print(command)
    # print()

# random.shuffle(command_list)

NUM_CHUNKS = len(datasets)

chunk_len = int(math.ceil(len(command_list) / NUM_CHUNKS))

for chunk_i in range(NUM_CHUNKS):
    start = chunk_i * chunk_len
    end = min((chunk_i + 1) * chunk_len, len(command_list))

    if start >= len(command_list):
        break

    if chunk_i == NUM_CHUNKS - 1:
        end = len(command_list)

    this_chunk_commands = command_list[start:end]
    print("; ".join(this_chunk_commands))
    print()
    print()
