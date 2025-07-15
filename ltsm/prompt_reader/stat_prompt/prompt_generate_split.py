# from ltsm.data_provider.data_factory import get_data_loader, get_data_loaders, get_dataset
import argparse
import ipdb
import pandas as pd
import numpy as np
import tsfel
from pandas import read_csv, read_feather
import matplotlib.pyplot as plt
import sys, os
import torch

# def parse_list(arg):
#     """parse a string of comma-separated values into a list
#     e.g. python ./ltsm/prompt_reader/stat_prompt/prompt_generate_split.py --dataset_name ETT-small, illness
#     """
#     return arg.split(',')

def get_args():
    parser = argparse.ArgumentParser(description='LTSM')

    parser.add_argument('--root_path', type=str, default='./datasets/', help='Root path for datasets')
    parser.add_argument('--output_path', type=str, default='./prompt_bank/stat-prompt/prompt_data_split/', help='Output path for prompt data')
    parser.add_argument('--dataset_name', nargs='+', default=[])
    parser.add_argument('--save_format', type=str, default='pth.tar',choices=["pth.tar", "csv", "npz"], help='The format to save the data')
    parser.add_argument('--test', type=bool, default=False)

    parser.add_argument('--data_path', type=str, default='dataset/weather.csv')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--freq', type=str, default="h")
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--percent', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--features', type=str, default='M')

    args = parser.parse_args()

    return args

def prompt_prune(pt):
    pt_dict = pt.to_dict()
    pt_keys = list(pt_dict.keys())
    for key in pt_keys:
        if key.startswith("0_FFT mean coefficient"):
            del pt[key]

    return pt


def prompt_generation_single(ts):
    """Generate prompt data for the input time-series data
    Args:
        ts (pd.Series): input time-series data
    """
    cfg = tsfel.get_features_by_domain()
    prompt = tsfel.time_series_features_extractor(cfg, ts)
    prompt = prompt_prune(prompt)
    return prompt

def prompt_generation(ts, ts_name):
    """Generate prompt data for the input time-series data
    Args:
        ts (pd.DataFrame): input time-series data
        ts_name (str): name of the time-series data
    """
    if ts.shape[1] == 1:
        return None

    else:
        column_name = [name.replace("/", "-") for name in list(ts.columns)]
        # column_name_map = {}
        # column_name = []
        # for i, name in enumerate(ts.columns):
        #     if not name.isnumeric():
        #         new_name = str(i)
        #     else:
        #         new_name = name
        #     column_name.append(new_name)
        #     column_name_map[name] = new_name
        prompt_buf_train = pd.DataFrame(np.zeros((133, ts.shape[1])), columns=column_name)
        prompt_buf_val = pd.DataFrame(np.zeros((133, ts.shape[1])), columns=column_name)
        prompt_buf_test = pd.DataFrame(np.zeros((133, ts.shape[1])), columns=column_name)
        for index, col in ts.T.iterrows():
            if "ETT" in ts_name:
                ts_len = len(ts)
                t1, t2 = int(0.6*ts_len), int(0.6*ts_len) + int(0.2*ts_len)
                ts_train, ts_val, ts_test = col[:t1], col[t1:t2].reset_index(drop=True), col[t2:].reset_index(drop=True)
            else:
                ts_len = len(ts)
                t1, t2 = int(0.7 * ts_len), int(0.7 * ts_len) + int(0.1 * ts_len)
                ts_train, ts_val, ts_test = col[:t1], col[t1:t2].reset_index(drop=True), col[t2:].reset_index(drop=True)

            prompt_train = prompt_generation_single(ts_train)
            prompt_val = prompt_generation_single(ts_val)
            prompt_test = prompt_generation_single(ts_test)

            prompt_buf_train[index.replace("/", "-")] = prompt_train.T.values
            prompt_buf_val[index.replace("/", "-")] = prompt_val.T.values
            prompt_buf_test[index.replace("/", "-")] = prompt_test.T.values
            # new_index = column_name_map[index]
            # prompt_buf_train[new_index] = prompt_train.T.values
            # prompt_buf_val[new_index] = prompt_val.T.values
            # prompt_buf_test[new_index] = prompt_test.T.values

    prompt_buf_total = {"train": prompt_buf_train, "val": prompt_buf_val, "test": prompt_buf_test}
    print(prompt_buf_total)
    return prompt_buf_total


def prompt_save(prompt_buf, output_path, data_name, save_format="pth.tar", ifTest=False):
    """save prompts to three different files in the output path
    Args:
        prompt_buf (dict): dictionary containing prompts for train, val, and test splits
        output_path (str): path to save the prompt data
        data_name (str): name of the dataset
        save_format (str): format to save the prompt data
        ifTest (bool): if True, test if the saved prompt data is loaded back. Can be used during generating data.
    """
    if prompt_buf["train"].shape[1] == 1:
        # ipdb.set_trace()
        return None

        # prompt_train_fname = os.path.join(prompt_train_data_dir, data_name + "_prompt.pth.tar")
        # prompt_train = prompt_buf["train"]
        # print("Export", prompt_train_fname, prompt_train.shape)
        #
        # prompt_val_fname = os.path.join(prompt_val_data_dir, data_name + "_prompt.pth.tar")
        # prompt_val = prompt_buf["val"]
        # torch.save(prompt_val, prompt_val_fname)
        # print("Export", prompt_val_fname, prompt_val.shape)
        #
        # prompt_test_fname = os.path.join(prompt_test_data_dir, data_name + "_prompt.pth.tar")
        # prompt_test = prompt_buf["test"]
        # torch.save(prompt_test, prompt_test_fname)
        # print("Export", prompt_test_fname, prompt_test.shape)

    else:
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(output_path, split)
            for index, col in prompt_buf[split].T.iterrows():
                file_name = f"{data_name}_{index}_prompt.{save_format}"
                file_path = os.path.join(split_dir, file_name)
                # print("split_dir", split_dir)
                # print("file_name", file_name)
                # print("file_path", file_path)
                prompt_data = col
                prompt_data.columns = [index]
                prompt_data = prompt_data.T
                print("Type of prompt data", type(prompt_data), "Shape of prompt data", prompt_data.shape)

                if save_format == "pth.tar":
                    torch.save(prompt_data, file_path)
                elif save_format == "csv":
                    prompt_data.to_csv(file_path, index=False)  # use csv may result in some loss of precision
                elif save_format == "npz":
                    np.savez(file_path, data=prompt_data.values, index=prompt_data.index, name=prompt_data.name)
                else:
                    raise ValueError(f"Unsupported save format: {save_format}")
                if ifTest:
                    if save_format == "pth.tar":
                        load_data = torch.load(file_path)
                    elif save_format == "csv":
                        load_data = pd.read_csv(file_path)
                        if isinstance(load_data, pd.DataFrame):
                            load_data = load_data.squeeze()
                    elif save_format == "npz":
                        loaded = np.load(file_path)
                        load_data = pd.Series(data=loaded["data"], index=loaded["index"], name=loaded["name"].item())
                        if isinstance(load_data, pd.DataFrame):
                            load_data = load_data.squeeze()
                    assert type(load_data) == type(prompt_data), f"Type mismatch: {type(load_data)} vs {type(prompt_data)}"  # type should be pd.Series
                    assert load_data.shape == prompt_data.shape, f"Shape mismatch: {load_data.shape} vs {prompt_data.shape}"
                    assert load_data.index.equals(prompt_data.index), "Index mismatch"
                    assert load_data.name == prompt_data.name, f"Series names mismatch: {load_data.name} vs {prompt_data.name}"
                    assert np.allclose(load_data.values, prompt_data.values, rtol=1e-8, atol=1e-8), "Data values mismatch"
                    if save_format != "csv":
                        assert load_data.equals(prompt_data), f"Data mismatch: {load_data} vs {prompt_data}"
                    print("All tests passed for", file_path)

                print("Export", file_path, prompt_data.shape)


def data_import(path, format="feather", anomaly=False):

    if format == "feather":
        data = read_feather(path)
        data_name = path.replace(root_path, "").replace(".feather", "")
        data_dir = data_name[0:data_name.rfind("/")]
        # ipdb.set_trace()
        data = data.value

    else:
        data = read_csv(path)
        data_name = path.replace(root_path, "").replace(".csv", "")
        data_dir = data_name[0:data_name.rfind("/")]
        if "date" in data.columns:
            data = data.drop("date", axis=1)
        if "anomaly" in data.columns:
            data = data.drop("anomaly", axis=1)
            print("Drop anomaly column")

    return data, data_name, data_dir


def create_data_dir(dir_name):
    # prompt_dir =
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == "__main__":

    args = get_args()
    root_path = args.root_path
    output_path = args.output_path
    dataset_name = args.dataset_name
    save_format = args.save_format
    ifTest = args.test    

    # if the dataset_name is not provided, use all the datasets in the dataset root path
    if not dataset_name:
        dataset_name = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]

    if len(dataset_name) == 0:
        print("No dataset found in the root path.")
        sys.exit(0)

    dataset_fullname = [os.path.join(root_path, name) for name in dataset_name]
    data_path_buf = []
    for dataset_dir in dataset_fullname:
        for root, dirs, files in os.walk(dataset_dir):
            for file_name in files:
                if file_name.endswith(".csv"):
                    file_path = os.path.join(root, file_name)
                    data_path_buf.append(file_path)

    print(data_path_buf)
    create_data_dir(output_path)
    # ipdb.set_trace()

    for path_idx, path in enumerate(data_path_buf):

        # print(path)

        data, data_name, data_dir = data_import(path, "csv")
        print("*****************Data Name: ", data_name)
        # print("Data Shape:", data.shape)
        if data.shape[0] < 20:
            print(path, "Skip too short time-series data.", data.shape)
            continue
        else:
            print("Import", path, "data shape", data.shape)

        create_data_dir(os.path.join(output_path, "train"))
        create_data_dir(os.path.join(output_path, "val"))
        create_data_dir(os.path.join(output_path, "test"))
        create_data_dir(os.path.join(output_path, "train", data_dir))
        create_data_dir(os.path.join(output_path, "val", data_dir))
        create_data_dir(os.path.join(output_path, "test", data_dir))

        prompt_data_buf = prompt_generation(data, data_name)
        if prompt_data_buf is not None:
            prompt_save(prompt_data_buf, output_path, data_name, save_format,ifTest)

