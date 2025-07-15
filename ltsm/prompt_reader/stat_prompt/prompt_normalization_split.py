# from ltsm.data_provider.data_factory import get_data_loader, get_data_loaders, get_dataset
import argparse
import ipdb
import pandas as pd
import numpy as np
#import tsfel
from pandas import read_csv, read_feather
import matplotlib.pyplot as plt
import sys, os
import torch
from sklearn.preprocessing import StandardScaler

# def parse_list(arg):
#     return arg.split(',')

def get_args():
    parser = argparse.ArgumentParser(description='LTSM')
    parser.add_argument('--mode', choices=["fit", "transform"], required=True)
    parser.add_argument('--dataset_name', nargs='+', default=[], help='The name of the dataset to be processed')
    parser.add_argument('--save_format', type=str, default='pth.tar', choices=["pth.tar", "csv", "npz"], help='The format to save the data')
    parser.add_argument('--root_path_train', type=str, default="./prompt_bank/stat-prompt/prompt_data_split/train", help='Root path for training data')
    parser.add_argument('--output_path_train', type=str, default="./prompt_bank/stat-prompt/prompt_data_normalize_split/train", help='Output path for normalized training data')
    parser.add_argument('--root_path_val', type=str, default="./prompt_bank/stat-prompt/prompt_data_split/val", help='Root path for validation data')
    parser.add_argument('--output_path_val', type=str, default="./prompt_bank/stat-prompt/prompt_data_normalize_split/val", help='Output path for normalized validation data')
    parser.add_argument('--root_path_test', type=str, default="./prompt_bank/stat-prompt/prompt_data_split/test", help='Root path for test data')
    parser.add_argument('--output_path_test', type=str, default="./prompt_bank/stat-prompt/prompt_data_normalize_split/test", help='Output path for normalized test data')
    parser.add_argument('--dataset_root', type=str, default="./datasets/", help='Output path for normalized test data')

    args = parser.parse_args()

    return args


# def prompt_generation(ts):
#     cfg = tsfel.get_features_by_domain()
#     prompt = tsfel.time_series_features_extractor(cfg, ts)
#     return prompt


def prompt_prune(pt):
    pt_dict = pt.to_dict()
    pt_keys = list(pt_dict.keys())
    for key in pt_keys:
        if type(key) == type("abc") and key.startswith("0_FFT mean coefficient"):
            del pt[key]

    return pt

def load_data(data_path, save_format):
    """load the prompt data in different format from the input path. This part is tested in tests/prompt_reader/test_prompt_generate_split.py
       The data should be pd.Series.
    Args:
        data_path: str, the input path
        save_format: str, the format of the data saved
    """
    if save_format == "pth.tar":
            prompt_data = torch.load(data_path)
    elif save_format == "csv":
        prompt_data = pd.read_csv(data_path)
        if isinstance(prompt_data, pd.DataFrame):
            prompt_data = prompt_data.squeeze()
    elif save_format == "npz":
        loaded = np.load(data_path)
        prompt_data = pd.Series(data=loaded["data"], index=loaded["index"], name=loaded["name"].item())
        if isinstance(prompt_data, pd.DataFrame):
            prompt_data = prompt_data.squeeze()
    return prompt_data

def save_data(data, data_path, save_format):
    """save the final prompt data to the output path
    Args:
        data: pd.DataFrame, the final prompt data
        data_path: str, the output path
        save_format: str, the format to save the data
    """
    if save_format == "pth.tar":
        torch.save(data, data_path)
    elif save_format == "csv":
        data.to_csv(data_path, index=False)
    elif save_format == "npz":
        np.savez(data_path, data=data.values, index=data.index, columns=data.columns) 

def mean_std_export_ds(data_path_buf, normalize_param_fname, save_format="pth.tar"):
    """Export the mean and std of the prompt data to the output path
    Args:
        data_path_buf: list, the list of the input path
        normalize_param_fname: str, the output path
        save_format: str, the format of the saved data
    """
    prompt_data_buf = []
    output_dir_buf = []
    output_path_buf = []
    for index, dataset_path in enumerate(data_path_buf):
        prompt_data = load_data(dataset_path, save_format)
        prompt_data = prompt_prune(prompt_data)
        prompt_data_buf.append(prompt_data)

        data_name = dataset_path.replace(root_path, "").replace(".csv", "")
        data_dir = data_name[0:data_name.rfind("/")]
        prompt_dir = os.path.join(output_path, data_dir)
        prompt_fname = os.path.join(output_path, data_name)
        # print(prompt_fname)
        output_dir_buf.append(prompt_dir)
        output_path_buf.append(prompt_fname)
        print("Import from {}".format(dataset_path), prompt_data.shape, type(prompt_data))
        # ipdb.set_trace()

    prompt_data_all = pd.concat(prompt_data_buf, axis=1).T
    print(prompt_data_all)

    scaler = StandardScaler()
    scaler.fit(prompt_data_all)

    sc_mean = pd.DataFrame(scaler.mean_.reshape(1,-1), columns=prompt_data_all.keys())
    sc_scale = pd.DataFrame(scaler.scale_.reshape(1,-1), columns=prompt_data_all.keys())

    print({"mean": sc_mean, "scale": sc_scale})
    print("Save the mean and std to {}".format(normalize_param_fname))
    torch.save({"mean": sc_mean, "scale": sc_scale}, normalize_param_fname)


def standardscale_export(data_path_buf, params_fname, output_path, root_path, save_format="pth.tar"):
    """Export the standardized prompt data to the output path
    Args:
        data_path_buf: list, the list of the input path
        params_fname: str, the output path of the mean and std
        output_path: str, the output path of the standardized prompt data
        root_path: str, the root path of the input"""
    params = torch.load(params_fname)
    print("Load from {}".format(params_fname), type(params))
    print(type(params["mean"]), type(params["scale"]))
    mean, std = params["mean"], params["scale"]
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = std
    # ipdb.set_trace()

    for index, dataset_path in enumerate(data_path_buf):
        prompt_data_raw = load_data(dataset_path, save_format)
        prompt_data_raw = prompt_prune(prompt_data_raw)

        prompt_data = scaler.transform(prompt_data_raw.values.reshape(1, -1))
        prompt_data_array = prompt_data
        # print(prompt_data)
        prompt_data_array[np.isnan(prompt_data_array)] = 0
        prompt_data_transform = pd.DataFrame(prompt_data_array, columns=prompt_data.keys())
        # ipdb.set_trace()

        prompt_fname = dataset_path.replace(root_path, output_path)
        prompt_dir = prompt_fname[0:prompt_fname.rfind("/")]
        if not os.path.exists(prompt_dir):
            os.makedirs(prompt_dir)
        # prompt_data_tramsform: pd.DataFrame,(1,133), column is RandeIndex
        # torch.save(prompt_data_transform, prompt_fname) 
        save_data(prompt_data_transform, prompt_fname, save_format)
        
        print("Save to {}".format(prompt_fname))
        del prompt_data


def create_data_dir(dir_name):
    # prompt_dir =
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

if __name__ == "__main__":
    args = get_args()
    
    ds_size = 50
    mode = args.mode # "transform" # "fit" #
    dataset_name = args.dataset_name
    save_format = args.save_format
    root_path_train = args.root_path_train
    output_path_train = args.output_path_train
    root_path_val = args.root_path_val
    output_path_val = args.output_path_val
    root_path_test = args.root_path_test
    output_path_test = args.output_path_test
    dataset_root_path = args.dataset_root

    print(dataset_name)

    if not dataset_name:
        dataset_name = [name for name in os.listdir(dataset_root_path) if os.path.isdir(os.path.join(dataset_root_path, name))]

    # since the params is a mid-state file, I didn't extend the file_format to the params file.
    data_path_buf = {
        "train": {"root_path": root_path_train, "output_path": output_path_train, "normalize_param_fname": os.path.join(output_path_train, f"normalization_params.pth.tar")},
        "val": {"root_path": root_path_val, "output_path": output_path_val, "normalize_param_fname": os.path.join(output_path_val, f"normalization_params.pth.tar")},
        "test": {"root_path": root_path_test, "output_path": output_path_test, "normalize_param_fname": os.path.join(output_path_test, f"normalization_params.pth.tar")},
    }

    for split_name, data_path in data_path_buf.items():
        root_path = data_path_buf[split_name]["root_path"]
        output_path = data_path_buf[split_name]["output_path"]
        normalize_param_fname = data_path_buf[split_name]["normalize_param_fname"]

        create_data_dir(output_path)

        dataset_fullname = [os.path.join(root_path, name) for name in dataset_name]
        data_path_buf_tmp = []
        if mode == "fit":

            for dataset_dir in dataset_fullname:
                paths = os.listdir(dataset_dir)
                new_dataset = [os.path.join(dataset_dir, path) for path in paths]
                sample_idx = np.random.permutation(len(new_dataset))[:ds_size].astype(np.int64)
                # ipdb.set_trace()
                new_dataset = np.array(new_dataset)[sample_idx].tolist()
                data_path_buf_tmp.extend(new_dataset)

        else:
            for dataset_dir in dataset_fullname:
                paths = os.listdir(dataset_dir)
                new_dataset = [os.path.join(dataset_dir, path) for path in paths]
                data_path_buf_tmp.extend(new_dataset)

        if mode == "fit":
            mean_std_export_ds(data_path_buf_tmp, normalize_param_fname, save_format)
        else:
            # ipdb.set_trace()
            standardscale_export(data_path_buf_tmp, normalize_param_fname, output_path, root_path, save_format)