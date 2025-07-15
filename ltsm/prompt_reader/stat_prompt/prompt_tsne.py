# from ltsm.data_provider.data_factory import get_data_loader, get_data_loaders, get_dataset
import argparse
import ipdb
import pandas as pd
import numpy as np
# import tsfel
from pandas import read_csv, read_feather
import matplotlib.pyplot as plt
import sys, os
import torch
from sklearn.preprocessing import StandardScaler
from sklearn import manifold


def get_args():
    parser = argparse.ArgumentParser(description='LTSM')

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


def prompt_generation(ts):
    cfg = tsfel.get_features_by_domain()
    prompt = tsfel.time_series_features_extractor(cfg, ts)
    return prompt


def prompt_prune(pt):
    pt_dict = pt.to_dict()
    pt_keys = list(pt_dict.keys())
    for key in pt_keys:
        if key.startswith("0_FFT mean coefficient"):
            del pt[key]

    return pt


if __name__ == "__main__":

    root_path = "./prompt_bank/stat-prompt/prompt_data_split/"
    # print(data_path_buf)

    dataset_name = [
        "electricity",
        "ETT-small",
        "exchange_rate",
        "illness",
        "traffic",
        "weather",
    ]
    split_buf = ["train", "val", "test"]

    dataset_fullname_train = [os.path.join(root_path, "train", name) for name in dataset_name]
    dataset_fullname_val = [os.path.join(root_path, "val", name) for name in dataset_name]
    dataset_fullname_test = [os.path.join(root_path, "test", name) for name in dataset_name]
    dataset_fullname = dataset_fullname_train + dataset_fullname_val + dataset_fullname_test
    data_path_buf = []
    dataset_dir_buf = []
    dataset_split_buf = []
    K = 100
    for index, dataset_dir in enumerate(dataset_fullname):
        paths = os.listdir(dataset_dir)
        new_dataset = [os.path.join(dataset_dir, path) for path in paths]
        sample_idx = np.random.permutation(len(new_dataset))[:K].astype(np.int64)
        # ipdb.set_trace()
        new_dataset = np.array(new_dataset)[sample_idx].tolist()
        data_path_buf.extend(new_dataset)

        for dataset_index, dname in enumerate(dataset_name):
            if dname in dataset_dir:
                dataset_dir_buf.extend(len(new_dataset) * [dataset_index])

        for split_index, split in enumerate(split_buf):
            if split in dataset_dir:
                dataset_split_buf.extend(len(new_dataset) * [split_index])
                break

    prompt_data_buf = []
    for index, dataset_path in enumerate(data_path_buf):
        prompt_data = torch.load(dataset_path)
        prompt_data_buf.append(prompt_data)
        print("Import from {}".format(dataset_path))
        # print(prompt_data)

        # if index == 100:
        #     break

    # print(prompt_data_buf)
    # print(output_path_buf)

    prompt_data_all = pd.concat(prompt_data_buf, axis=0).values
    print(prompt_data_all.shape)
    # (3166, 133)
    
    # nan_index = np.where(np.isnan(prompt_data_all))[0]
    # prompt_data_all[nan_index] = 0

    # ipdb.set_trace()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    prompt_data_tsne = tsne.fit_transform(prompt_data_all)
    dataset_plot_buf = ["electricity"]
    color_buf = ["red", "blue", "black", "green", "pink", "brown"]
    marker_buf = [".", "^", "x"]
    for index, _ in enumerate(dataset_name):
        for sindex, split_fold in enumerate(split_buf):
            data_index = (np.array(dataset_dir_buf) == index)
            split_index = (np.array(dataset_split_buf) == sindex)
            plot_index = data_index & split_index
            plt.plot(prompt_data_tsne[plot_index, 0], prompt_data_tsne[plot_index, 1], linewidth=0, marker=marker_buf[sindex], label=str(dataset_name[index][0:8] + "-" + split_fold), color=color_buf[index])
            # plt.text(prompt_data_tsne[data_index, 0].mean()-20, prompt_data_tsne[data_index, 1].mean(), str(dataset_name[index][0:8]), fontdict={'weight': 'bold', 'size': 9})

    plt.legend(loc="right")
    plt.savefig("./figures/stat_prompt_tsne.png")
    plt.close()

    # ipdb.set_trace()
    # plt.xticks([])
    # plt.yticks([])

    # print(prompt_data_all)
    # , color = plt.cm.Set1(dataset_dir_buf[index])
    # print(prompt_data_transform)
    # print(prompt_data_transform_array.mean(axis=0))
    # print(prompt_data_transform_array.std(axis=0))
    # print(prompt_data_transform.loc[5])






