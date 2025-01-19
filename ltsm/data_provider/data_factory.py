import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from ltsm.data_reader import reader_dict
from ltsm.data_provider.data_splitter import SplitterByTimestamp
from ltsm.data_provider.tokenizer import processor_dict
from ltsm.data_provider.dataset import TSDataset,  TSPromptDataset, TSTokenDataset
from ltsm.data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute

from typing import Tuple, List, Union, Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

class DatasetFactory:
    """
    A factory class for time-series datasets.
    """
    def __init__(
        self, 
        data_paths: List[str], 
        prompt_data_path: str, 
        data_processing: str, 
        seq_len: int, 
        pred_len: int, 
        train_ratio: float, 
        val_ratio: float,
        model: str= None,
        scale_on_train: bool = False, 
        downsample_rate: int = 10,
        split_test_sets: bool = True,
        do_anomaly: bool = False
    ):
        """
        Initializes the DatasetFactory with the given arguments.

        Args:
            data_paths (List[str]): A list of file paths where the source data is stored.
            prompt_data_path (str): The file path to the prompt data folder.
            data_processing (str): The module ID of the processor in processor_dict.
            seq_len (int): The number of timesteps used in the input sequence.
            pred_len (int): The number of timesteps the model should predict for the output sequence.
            train_ratio (float): The training set ratio.
            val_ratio (float): The validation set ratio.
            model (str): The model name. Options includes 'LTSM', 'LTSM_WordPrompt', and 'LTSM_Tokenizer'.
            scale_on_train (bool): Indicates whether the datasets should be scaled based on the training data.
            downsample_rate (int): The downsampling rate for training and validation datasets.
            split_test_sets (bool): Indicates whether the test sets should be saved separately by data_path.
        """
        self.data_paths = data_paths
        self.prompt_data_path = prompt_data_path
        self.model = model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale_on_train = scale_on_train
        self.downsample_rate = downsample_rate
        self.split_test_sets = split_test_sets
        self.do_anomaly = do_anomaly

        # Initialize dataset splitter
        self.splitter = SplitterByTimestamp(
            seq_len,
            pred_len,
            train_ratio,
            val_ratio
        )

        # Initialize the data preprocessor
        self.processor = processor_dict[data_processing]()

    def fetch(self, data_path: str)->pd.DataFrame:
        """
        Retrieves data from the filesystem at the specified data_path. 

        Selects the appropriate BaseReader implementation based on the file's extension or location.

        Args:
            data_path (str): The file path to the source data.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the data at data_path.
        """
        # If data path is in monash directory, use monash reader
        dir_name = os.path.split(os.path.dirname(data_path))[-1]
        if dir_name == 'monash':
            return reader_dict[dir_name](data_path).fetch()
        
        # Get file extension
        ext = os.path.splitext(data_path)[-1]
        return reader_dict[ext[1:]](data_path).fetch()
    
    def __get_prompt(self, prompt_data_path:str, data_name: str, idx_file_name: str) -> List[np.float64]:
        """
            Private helper function to load prompt data files.

            Args:
                prompt_data_path (str): The path to the directory where the prompt data files are stored.
                data_name (str): The name of the data source.
                idx_file_name (str): The row label corresponding to the data the prompt file was generated from.

            Returns:
                List[np.float64]: The raw prompt data.
        """
        # Prompt file name replaces '/' in row labels with '-' 
        idx_file_name = idx_file_name.replace("/", "-")

        # Certain characters cannot be used in file names
        idx_file_name = idx_file_name.replace("**", "_")
        idx_file_name = idx_file_name.replace("%", "_")
        
        if os.path.split(os.path.dirname(data_name))[-1] == "monash":
            # Monash
            prompt_name = data_name.split("/")[-1]
            prompt_name = prompt_name.replace(".tsf", "")
            prompt_path = os.path.join(prompt_data_path, prompt_name, "T"+str(int(idx_file_name)+1)+"_prompt")
        else:
            # CSV and other
            prompt_name = data_name.split('/')[-2]+'/'+data_name.split('/')[-1].split('.')[0]
            prompt_path = os.path.join(prompt_data_path,prompt_name+'_'+str(idx_file_name)+"_prompt")

        try:
            # Check for the existence of the prompt file in different formats
            if os.path.exists(prompt_path + '.csv'):
                prompt_path += '.csv'
                print(f"Prompt file {prompt_path} exists")
                prompt_data = pd.read_csv(prompt_path)
                prompt_data.columns = prompt_data.columns.astype(int)
            elif os.path.exists(prompt_path + '.pth.tar'):
                prompt_path += '.pth.tar'
                prompt_data = torch.load(prompt_path)  
            elif os.path.exists(prompt_path + '.npz'):
                prompt_path += '.npz'
                loaded_data = np.load(prompt_path)
                prompt_data = pd.DataFrame(loaded_data['data']) # this should match the key saved in prompt_generate_split.py
            else:
                logging.error(f"Prompt file {prompt_path} does not exist in any supported format")
                return []
            # after load the data, it should be (1, 133). 133 is decided in prompt_generate_split.py
            prompt_data = prompt_data.T[0]  # should be (133,)
            prompt_data = [ prompt_data.iloc[i] for i in range(len(prompt_data)) ]
        except Exception as e:
            logging.error(e)
            return []
        
        return prompt_data

    
    def loadPrompts(self, data_path: str, prompt_data_path:str, buff: List[Union[int, str]])->Tuple[Dict[Union[int, str], List[np.float64]], List[Union[int, str]]]:
        """
        Loads the prompt data from prompt_data_path.

        Args:
            data_path (str): The file path to the source data.
            prompt_data_path (str): The file path to the directory where the prompt data files are stored.
            buff (List[Union[int, str]]): The list of row labels of the data.

        Returns:
            Dict[Union[int, str], List[np.float64]]: A dictionary of data indices and the prompt data corresponding to each index.
            List[Union[int, str]]: A list of indices with missing prompt data
        """
        prompt_data = []
        missing = []
        if "WordPrompt" in self.model:
            # Load index of every data class for each instance, as prompt data will be different for different datasets
            for _ in buff:
                prompt_data.append([self.data_paths.index(data_path)])
        else:
            for instance_idx in buff:
                if instance_idx == 'anomaly':
                    continue # anomaly is the label for anomaly data, no prompt data is available
                instance_prompt = self.__get_prompt(
                    prompt_data_path,  
                    data_path,
                    str(instance_idx)
                )
                # If no prompt is loaded for this instance, save instance_idx to missing list
                if len(instance_prompt) == 0:
                    missing.append(instance_idx)
                    logging.info(f"Prompt data for index {instance_idx} of data path {data_path} cannot be read. Skipping this row.")

                prompt_data.append(instance_prompt)

        return prompt_data, missing
    
    def createTorchDS(self, data: List[np.ndarray], prompt_data: List[List[np.float64]], downsample_rate: int, do_anomaly:bool)->TSDataset:
        """
        Creates a pyTorch Dataset from a list of sequences and a list of their corresponding prompts.

        Args:
            data (List[np.ndarray]): A list of sequences.
            prompt_data (List[List[np.float64]]): A list of prompts.
            downsample_rate: The downsampling rate.

        Returns:
            TSDataset: A time-series dataset.
        """
        if len(data) == 0 or len(prompt_data) == 0:
            return None
        
        if "Tokenizer" in self.model:
            return TSTokenDataset(
                data=data,
                prompt=prompt_data,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                downsample_rate=downsample_rate,
                do_anomaly=do_anomaly
            )
        else:
            return TSPromptDataset(
                data=data,
                prompt=prompt_data,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                downsample_rate=downsample_rate,
                do_anomaly=do_anomaly
            )

    def getDatasets(self)->Tuple[TSDataset, TSDataset, List[TSDataset]]:
        """
        Loads, splits, and sclaes the time-series data. Loads the prompts and creates TSDatasets for training, validation,
        and testing. 

        Returns:
            Tuple[TSDataset, TSDataset, List[TSDataset]]:
                A tuple consisting of the time-series datasets for training, validation, and testing.
                The training and validation datasets combine all data sources and sequences into a single dataset, respectively.
                Test data is kept separate and are returned as a list of time-series datasets where each dataset corresponds to 
                one of the data sources.
        """
        train_data, val_data, test_data, train_prompt_data, val_prompt_data, test_prompt_data = [], [], [], [], [], []
        test_ds_list = []
        for data_path in self.data_paths:
            # Step 0: Read data, the output is a list of 1-d time-series
            df_data = self.fetch(data_path)
            #print(df_data.index)
            #print(df_data.loc['anomaly'].iloc[:20])
            #print(df_data.shape)

            # Step 1: Get train, val, and test splits
            sub_train_data, sub_val_data, sub_test_data, buff = self.splitter.get_csv_splits(df_data, self.do_anomaly)
            # print(len(sub_train_data), len(sub_val_data), len(sub_test_data))
            # print(sub_train_data[0].shape, sub_val_data[0].shape, sub_test_data[0].shape)
            #print(sub_train_data[-1][:10])
            

            # Step 2: Scale the datasets. We fit on the whole sequence by default.
            # To fit on the train sequence only, set scale_on_train=True
            sub_train_data, sub_val_data, sub_test_data = self.processor.process(
                raw_data=df_data.to_numpy(),
                train_data=sub_train_data,
                val_data=sub_val_data,
                test_data=sub_test_data,
                fit_train_only=self.scale_on_train,
                do_anomaly=self.do_anomaly
            )
            #print(type(sub_train_data), type(sub_val_data), type(sub_test_data))
            logging.info(f"Data {data_path} has been split into train, val, test sets with the following shapes: {sub_train_data[0].shape}, {sub_val_data[0].shape}, {sub_test_data[0].shape}")

            # Step 2.5: Load prompt for each instance
            missing = set()
            # Train Prompt
            train_prompt_data_path = self.prompt_data_path + '/train'
            train_prompts, tmp = self.loadPrompts(data_path, train_prompt_data_path, buff)
            missing.update(set(tmp))
            
            
            # Validation Prompt
            val_prompt_data_path = self.prompt_data_path + '/val'
            val_prompts, tmp = self.loadPrompts(data_path, val_prompt_data_path, buff)
            missing.update(set(tmp))
            

            # Test Prompt
            test_prompt_data_path = self.prompt_data_path + '/test'
            sub_test_prompt_data, tmp = self.loadPrompts(data_path, test_prompt_data_path, buff)
            missing.update(set(tmp))

            # Remove rows from data that have missing prompt data
            sub_train_data = [data for data, instance_idx in zip(sub_train_data, buff) if instance_idx not in missing]
            sub_val_data = [data for data, instance_idx in zip(sub_val_data, buff) if instance_idx not in missing]
            sub_test_data = [data for data, instance_idx in zip(sub_test_data, buff) if instance_idx not in missing]

            # Only maintain instances that are able to load prompt data for training, validation, and testing
            train_prompts = [data for data, instance_idx in zip(train_prompts, buff) if instance_idx not in missing]
            val_prompts = [data for data, instance_idx in zip(val_prompts, buff) if instance_idx not in missing]
            sub_test_prompt_data = [data for data, instance_idx in zip(sub_test_prompt_data, buff) if instance_idx not in missing]

            # print(len(sub_train_data), len(sub_val_data), len(sub_test_data))
            # print(sub_train_data[0].shape, sub_val_data[0].shape, sub_test_data[0].shape)

            if self.do_anomaly:
                label_train = sub_train_data[-1]
                label_val = sub_val_data[-1]
                label_test = sub_test_data[-1]

                sub_train_data = [[(x,y) for x,y in zip(data, label_train)] for data in sub_train_data[:-1]]
                sub_val_data = [[(x,y) for x,y in zip(data, label_val)] for data in sub_val_data[:-1]]
                sub_test_data = [[(x,y) for x,y in zip(data, label_test)] for data in sub_test_data[:-1]]

                # print(len(sub_train_data), len(sub_val_data), len(sub_test_data))
                # print(sub_train_data[0][0:2], sub_val_data[0][0:2], sub_test_data[0][0:2])
                

            train_prompt_data.extend(train_prompts)
            val_prompt_data.extend(val_prompts)
            
            train_data.extend(sub_train_data)
            val_data.extend(sub_val_data)

            


            if self.split_test_sets:
                # Create a Torch dataset for each sub test dataset
                test_ds_list.append(self.createTorchDS(sub_test_data, sub_test_prompt_data, 1, self.do_anomaly))
            else:
                test_data.extend(sub_test_data)
                test_prompt_data.extend(sub_test_prompt_data)
        
        # Step 3: Create Torch datasets (samplers)
        # print("====================================")
        # print("len(train_data):", len(train_data))
        # print("len(train_data[0]):", len(train_data[0]),len(train_data[1]))
        # print("len(train_data[-1]):",len(train_data[-1]),len(train_data[-2]))
        # print("====================================")
        
        train_ds = self.createTorchDS(train_data, train_prompt_data, self.downsample_rate, self.do_anomaly)
        if os.path.split(os.path.dirname(self.data_paths[0]))[-1] == "monash":
            val_ds = self.createTorchDS(val_data, val_prompt_data, 54, self.do_anomaly)
        else:
            val_ds = self.createTorchDS(val_data, val_prompt_data, self.downsample_rate, self.do_anomaly)

        if not self.split_test_sets:
            test_ds_list.append(self.createTorchDS(test_data, test_prompt_data, 1, self.do_anomaly))
        
        return train_ds, val_ds, test_ds_list

def get_datasets(args): 
    if "LTSM" in args.model:
        # Create datasets
        dataset_factory = DatasetFactory(
            data_paths=args.data_path,
            prompt_data_path=args.prompt_data_path,
            data_processing=args.data_processing,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            model=args.model,
            split_test_sets=False,
            downsample_rate=args.downsample_rate,
            do_anomaly=args.do_anomaly
        )
        train_dataset, val_dataset, test_datasets = dataset_factory.getDatasets()
        processor = dataset_factory.processor
    else:
        timeenc = 0 if args.embed != 'timeF' else 1
        Data = Dataset_Custom
        if args.data == "ETTh1" or args.data == "ETTh2":
            Data = Dataset_ETT_hour
        elif args.data == "ETTm1" or args.data == "ETTm2":
            Data = Dataset_ETT_minute

        train_dataset = Data(
            data_path=args.data_path[0],
            split='train',
            size=[args.seq_len, args.pred_len],
            freq=args.freq,
            timeenc=timeenc,
            features=args.features
        )
        val_dataset = Data(
            data_path=args.data_path[0],
            split='val',
            size=[args.seq_len, args.pred_len],
            freq=args.freq,
            timeenc=timeenc,
            features=args.features
        )
        test_datasets = [Data(
            data_path=args.data_path[0],
            split='test',
            size=[args.seq_len, args.pred_len],
            freq=args.freq,
            timeenc=timeenc,
            features=args.features
        )]
        processor = train_dataset.scaler

    return train_dataset, val_dataset, test_datasets, processor
    
def get_data_loaders(args):
    train_dataset, val_dataset, test_datasets, processor = get_datasets()
    print(f"Data loaded, train size {len(train_dataset)}, val size {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # split_test_data set to False, length of test_datasets is 1
    test_loader = DataLoader(
        test_datasets[0],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader, processor