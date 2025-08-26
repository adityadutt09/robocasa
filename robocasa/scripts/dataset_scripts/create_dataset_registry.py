import os
from collections import OrderedDict
from pathlib import Path
import h5py
import numpy as np
import json
from tqdm import tqdm
import robocasa
import robocasa.macros as macros


def get_valid_paths(base_path):
    ds_paths = []
    print(f"Scanning for datasets in {base_path} ...")

    start_path_depth = base_path.count(os.sep)
    for root, dirs, files in os.walk(base_path):
        current_depth = root.count(os.sep) - start_path_depth

        if current_depth >= 4:
            del dirs[:]  # Prune subdirectories to prevent further recursion

        for f in files:
            if f == "demo.hdf5":
                ds_paths.append(os.path.join(root, f))

    return ds_paths


# scan for datasets
ds_base_path = macros.DATASET_BASE_PATH or os.path.join(
    os.path.dirname(robocasa.__path__[0]), "datasets"
)
v05_base_path = os.path.join(ds_base_path, "v0.5")

all_ds_paths = []
for base_path in [
    os.path.join(v05_base_path, "train", "atomic"),
    os.path.join(v05_base_path, "test", "atomic"),
    # os.path.join(v05_base_path, "train", "composite"),
    # os.path.join(v05_base_path, "test", "composite"),
]:
    all_ds_paths += get_valid_paths(base_path)

all_ds_paths.sort()
all_ds_paths = all_ds_paths[::-1]  # go in reverse chronological order

print("Reading configs...")
ds_configs = {}
for ds_path in tqdm(all_ds_paths):
    is_mg = "/mg/" in ds_path
    split = "train" if "/train/" in ds_path else "test"

    # get the env name
    try:
        f = h5py.File(ds_path)
        # env_name = f["data"].attrs["env"]
        env_args = json.loads(f["data"].attrs["env_args"])
        env_name = env_args["env_name"]
    except Exception as e:
        print("Exception reading", ds_path)
        continue

    if env_name not in ds_configs:
        ds_configs[env_name] = dict(
            train=dict(),
            test=dict(),
        )

    if is_mg:
        if "mg_path" in ds_configs[env_name][split]:
            print("mg path already detected!", ds_path)
            continue
    else:
        if "human_path" in ds_configs[env_name][split]:
            print("human path already detected!", ds_path)
            continue

    # only include datasets that have at least 90 trajectories
    try:
        if len(list(f["data"].keys())) < 90:
            continue
    except Exception as e:
        print("Exception reading...")
        f.close()
        continue

    rel_dir = os.path.relpath(os.path.dirname(ds_path), ds_base_path)

    if is_mg:
        ds_configs[env_name][split]["mg_path"] = rel_dir
    else:
        ds_configs[env_name][split]["human_path"] = rel_dir

        # get the traj_lengths
        try:
            demos = sorted(list(f["data"].keys()))
            traj_lengths = []
            for ep in demos:
                traj_lengths.append(f["data/{}/actions".format(ep)].shape[0])
            traj_lengths = np.array(traj_lengths)
        except Exception as e:
            print("Exception reading...")
            f.close()
            continue

        # compute dataset horizon
        mean, std = np.mean(traj_lengths), np.std(traj_lengths)
        # round to next hundred steps
        horizon = int(((mean + 2 * std) // 100 * 100) + 100)

        if "horizon" in ds_configs[env_name]:
            ds_configs[env_name]["horizon"] = max(
                horizon, ds_configs[env_name]["horizon"]
            )
        else:
            ds_configs[env_name]["horizon"] = horizon

    # close the hdf5 file at the end
    f.close()

print()
print()
print()
for env_name in sorted(ds_configs.keys()):
    cfg = ds_configs[env_name]
    text = f"{env_name}=dict(\n"
    for k, v in cfg.items():
        if isinstance(v, dict):
            if len(v) == 0:
                # skip empty dictionaries
                continue
            v_str = f"dict(\n"
            for k1, v1 in v.items():
                v1_str = repr(v1).replace("'", '"')
                v_str += f"        {k1}={v1_str},\n"
            v_str += f"    )"
        else:
            v_str = repr(v).replace("'", '"')
        text += f"    {k}={v_str},\n"
    text += "),"
    print(text)
