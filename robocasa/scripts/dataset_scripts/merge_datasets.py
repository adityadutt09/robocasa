from robomimic.scripts.internal.merge_hdf5 import merge_hdf5s
import h5py
from tqdm import tqdm
import os

import robocasa
import robocasa.macros as macros
import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils

import argparse


def merge_datasets(source_datasets, output_dataset):
    print(f"Merging to {output_dataset}")

    with h5py.File(output_dataset, "w") as f_out:
        data_grp = f_out.create_group("data")
        total_samples = 0

        merged_demo_num = 1

        # Copy data maintaining original demo order by following the original demos list
        pbar = tqdm(source_datasets, desc="Merging")
        try:
            for dataset in pbar:
                # Look up which temp file contains this demo
                with h5py.File(dataset, "r") as f_dataset:
                    for demo in tqdm(f_dataset["data"]):
                        # Copy this demo's data to the output file
                        merged_demo_name = f"demo_{merged_demo_num}"

                        f_dataset.copy(f"data/{demo}", data_grp, name=merged_demo_name)
                        total_samples += data_grp[merged_demo_name].attrs["num_samples"]
                        # Copy environment args from the first temp file if not done yet
                        if "env_args" not in data_grp.attrs:
                            data_grp.attrs["env_args"] = f_dataset["data"].attrs[
                                "env_args"
                            ]

                        merged_demo_num += 1

                pbar.set_postfix({"total_samples": total_samples})
        finally:
            pbar.close()

        data_grp.attrs["total"] = total_samples

    DatasetUtils.extract_action_dict(dataset=output_dataset)
    DatasetUtils.make_demo_ids_contiguous(dataset=output_dataset)
    for num_demos in [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        75,
        80,
        90,
        100,
        125,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        1500,
        2000,
        2500,
        3000,
        4000,
        5000,
        10000,
    ]:
        DatasetUtils.filter_dataset_size(
            output_dataset,
            num_demos=num_demos,
        )


def find_mg_datasets(task):
    dataset_base_path = macros.DATASET_BASE_PATH
    if dataset_base_path is None:
        dataset_base_path = os.path.join(robocasa._path__[0], "..", "datasets")
    mg_base_path = os.path.join(
        dataset_base_path, "v0.5/train/atomic", task, "mg", "demo"
    )
    all_timestamps = []
    for timestamp in os.listdir(mg_base_path):
        if not os.path.isdir(os.path.join(mg_base_path, timestamp)):
            continue
        all_timestamps.append(timestamp)
    latest_timestamp = sorted(all_timestamps)[-1]

    mg_run_path = os.path.join(mg_base_path, latest_timestamp)

    source_datasets = []
    for file in os.listdir(mg_run_path):
        if "demo.hdf5_temp_" in file:
            source_datasets.append(os.path.join(mg_run_path, file))

    output_dataset = os.path.join(mg_run_path, "demo_im128.hdf5")

    return source_datasets, output_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="name of the task to merge together",
    )
    args = parser.parse_args()

    source_datasets, output_dataset = find_mg_datasets(task=args.task)
    # source_datasets = [
    #     '/mnt/amlfs-01/shared/robocasa_benchmark/robocasa-datasets/v0.5/train/atomic/OpenCabinet/mg/demo/2025-07-29-14-56-25_and_2025-08-01-23-36-13/demo1.hdf5',
    #     '/mnt/amlfs-01/shared/robocasa_benchmark/robocasa-datasets/v0.5/train/atomic/OpenCabinet/mg/demo/2025-07-29-14-56-25_and_2025-08-01-23-36-13/demo2.hdf5',
    # ]
    # output_dataset = '/mnt/amlfs-01/shared/robocasa_benchmark/robocasa-datasets/v0.5/train/atomic/OpenCabinet/mg/demo/2025-07-29-14-56-25_and_2025-08-01-23-36-13/demo.hdf5'
    merge_datasets(source_datasets, output_dataset)
