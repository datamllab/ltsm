import ipdb
import pandas as pd
import numpy as np
from pandas import read_csv, read_feather
import sys, os
import torch
from sklearn.preprocessing import StandardScaler

# Add the path to `tsfel` dynamically
tsfel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../prompt_reader/stat_prompt"))
sys.path.append(tsfel_path)
import tsfel

def prompt_prune(pt):
    pt_dict = pt.to_dict()
    pt_keys = list(pt_dict.keys())
    for key in pt_keys:
        if type(key) == type("abc") and key.startswith("0_FFT mean coefficient"):
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


def data_import(path, root_path, format="feather", anomaly=False):

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
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def prompt_generate_split(root_path: str, output_path: str, save_format:str, dataset_name: str = None, ifTest=False) -> None:
    """Generate prompt data for the input time-series data
    Args:
        root_path (str): path to the dataset
        output_path (str): path to save the prompt data
        save_format (str): format to save the prompt data
        dataset_name (str): name of the dataset
        ifTest (bool): if True, test if the saved prompt data is loaded back. Can be used during generating data.
    """
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

        data, data_name, data_dir = data_import(path, root_path, "csv")
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

        print(data)

        prompt_data_buf = prompt_generation(data, data_name)
        if prompt_data_buf is not None:
            prompt_save(prompt_data_buf, output_path, data_name, save_format, ifTest)

def load_data(data_path, save_format):
    """Load the prompt data in different format from the input path.
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
    """Save the final prompt data to the output path
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

def mean_std_export_ds(root_path, output_path, data_path_buf, normalize_param_fname, save_format="pth.tar"):
    """Export the mean and std of the prompt data to the output path
    Args:
        root_path: str, the root path of the input
        output_path: str, the output path
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

def prompt_normalization_split(mode: str, save_format: str, root_path_train: str, output_path_train: str, root_path_val: str, 
                               output_path_val: str, root_path_test: str, output_path_test: str, dataset_root_path: str, dataset_name: str = None) -> None:
    """Normalize the prompt data for the input time-series data
    Args:
        mode (str): mode to run, "fit" or "transform"
        save_format (str): format to save the prompt data
        root_path_train (str): path to the train dataset
        output_path_train (str): path to save the train prompt data
        root_path_val (str): path to the val dataset
        output_path_val (str): path to save the val prompt data
        root_path_test (str): path to the test dataset
        output_path_test (str): path to save the test prompt data
        dataset_root_path (str): path to the dataset root
        dataset_name (str): name of the dataset
    """
    ds_size = 50
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
            mean_std_export_ds(root_path, output_path, data_path_buf_tmp, normalize_param_fname, save_format)
        else:
            # ipdb.set_trace()
            standardscale_export(data_path_buf_tmp, normalize_param_fname, output_path, root_path, save_format)