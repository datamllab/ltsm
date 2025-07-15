import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

from ltsm.data_provider.tokenizer.tokenizer_processor import TokenizerConfig

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

class TSDataset(Dataset):
    def __init__(
        self, 
        data, 
        seq_len,
        pred_len,
        do_anomaly=False,
    ):
        self.data = data
        self.seq_len = seq_len 
        self.do_anomaly = do_anomaly

        if not self.do_anomaly:
            self.pred_len = pred_len
        else:
            self.pred_len = self.seq_len

        # Create a map from item index to sequence index and offset
        self.num_items = 0
        self.item2sequence, self.item2offset = [], []
        
        for sequence_index, sequence in enumerate(self.data):
            if not self.do_anomaly:
                assert len(sequence) >= self.seq_len + self.pred_len, f"Sequence must have a lenth with at least seq_len + pred_len, the current length is {len(sequence)}"
                window_size = len(sequence) - self.seq_len - self.pred_len + 1
            else:
                window_size = len(sequence) - self.seq_len + 1
            cur_offset = 0
            for _ in range(window_size):
                self.item2sequence.append(sequence_index)
                self.item2offset.append(cur_offset)
                cur_offset += 1
                self.num_items += 1

    def __getitem__(self, index):
        if not self.do_anomaly:
            sequence_index = self.item2sequence[index]
            x_begin = self.item2offset[index]
            x_end = x_begin + self.seq_len
            y_begin = x_end
            y_end = y_begin + self.pred_len 
            
            seq_x = torch.from_numpy(np.expand_dims(self.data[sequence_index][x_begin:x_end], -1))
            seq_y = torch.from_numpy(np.expand_dims(self.data[sequence_index][y_begin:y_end], -1))
        else:
            sequence_index = self.item2sequence[index]
            x_begin = self.item2offset[index]
            x_end = x_begin + self.seq_len
            y_begin = x_begin
            y_end = x_end

            data_x = np.array([x for x,y in self.data[sequence_index][x_begin:x_end]])
            data_y = np.array([y for x,y in self.data[sequence_index][y_begin:y_end]])

            seq_x = torch.from_numpy(np.expand_dims(data_x, -1))
            seq_y = torch.from_numpy(np.expand_dims(data_y, -1))

        return seq_x, seq_y

    def __len__(self):
        return self.num_items
    
class TSPromptDataset(Dataset):
    def __init__(
        self, 
        data, 
        prompt,
        seq_len,
        pred_len,
        downsample_rate=10,
        do_anomaly=False,
    ):
        self.prompt = prompt
        self.seq_len = seq_len 
        if not do_anomaly:
            self.pred_len = pred_len
        else:
            self.pred_len = self.seq_len
        self.num_items = 0
        self.item2sequence, self.item2offset = [], []
        self.data  = data
        self.do_anomaly = do_anomaly

        for sequence_index, sequence in enumerate(self.data):
            if not self.do_anomaly:
                assert len(sequence) >= self.seq_len + self.pred_len, f"Sequence must have a lenth with at least seq_len + pred_len, the current length is {len(sequence)}"
                window_size = len(sequence) - self.seq_len - self.pred_len + 1
            else:
                window_size = len(sequence) - self.seq_len + 1
            cur_offset = 0
            for _ in range(window_size):
                self.item2sequence.append(sequence_index)
                self.item2offset.append(cur_offset)
                cur_offset += 1
                self.num_items += 1
            
            

    def __getitem__(self, index):
        if not self.do_anomaly:
            sequence_index = self.item2sequence[index]
            x_begin = self.item2offset[index]
            x_end = x_begin + self.seq_len
            y_begin = x_end
            y_end = y_begin + self.pred_len
            prompt= self.prompt[sequence_index]
            
            # prompt is a list, self.data[sequence_index][x_begin:x_end])is a numpy array with shape(seq_len,), like (366,)
            seq_x = np.concatenate((prompt, self.data[sequence_index][x_begin:x_end])) 
            seq_x = torch.from_numpy(np.expand_dims(seq_x, -1))
            seq_y = torch.from_numpy(np.expand_dims(self.data[sequence_index][y_begin:y_end], -1))
        else:
            sequence_index = self.item2sequence[index]
            x_begin = self.item2offset[index]
            x_end = x_begin + self.seq_len
            y_begin = x_begin
            y_end = x_end
            prompt= self.prompt[sequence_index]

            data_x = np.array([x for x,y in self.data[sequence_index][x_begin:x_end]])
            data_y = np.array([y for x,y in self.data[sequence_index][y_begin:y_end]])

            seq_x = np.concatenate((prompt, data_x))
            seq_x = torch.from_numpy(np.expand_dims(seq_x, -1))
            seq_y = torch.from_numpy(np.expand_dims(data_y, -1))

        return seq_x, seq_y

    def __len__(self):
        return self.num_items

class TSTokenDataset(Dataset):
    def __init__(
        self, 
        data, 
        prompt,
        seq_len,
        pred_len,
        downsample_rate=10,
        do_anomaly=False,
    ):
        self.seq_len = seq_len 
        if not do_anomaly:
            self.pred_len = pred_len
        else:
            self.pred_len = self.seq_len
        self.num_items = 0
        self.item2sequence, self.item2offset = [], []
        self.data  = data
        self.prompt = prompt

        for sequence_index, sequence in enumerate(self.data):
            assert len(sequence) >= self.seq_len + self.pred_len, f"Sequence must have a length with at least seq_len + pred_len, the current length is {len(sequence)}"
            cur_offset = 0
            for cur_offset in range(0, len(sequence) - self.seq_len - self.pred_len + 1, downsample_rate):
                self.item2sequence.append(sequence_index)
                self.item2offset.append(cur_offset)
                self.num_items += 1
            
        context_length = seq_len+pred_len
        prediction_length = pred_len
        n_tokens = 1024
        n_special_tokens = 2
        config = TokenizerConfig(
            tokenizer_class="MeanScaleUniformBins",
            tokenizer_kwargs=dict(low_limit=-3.0, high_limit=3.0),
            n_tokens=n_tokens,
            n_special_tokens=n_special_tokens,
            pad_token_id=0,
            eos_token_id=1,
            use_eos_token=0,
            model_type="causal",
            context_length=context_length,
            prediction_length=prediction_length,
            num_samples=20,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
        )

        self.tokenizer = config.create_tokenizer()

        for sequence_index, sequence in enumerate(self.data):
            assert len(sequence) >= self.seq_len + self.pred_len, f"Sequence must have a length with at least seq_len + pred_len, the current length is {len(sequence)}"
            cur_offset = 0
            for cur_offset in range(0, len(sequence) - self.seq_len - self.pred_len + 1, downsample_rate):
                self.item2sequence.append(sequence_index)
                self.item2offset.append(cur_offset)
                # cur_offset += 1
                self.num_items += 1
            
        
    def __getitem__(self, index):
        sequence_index = self.item2sequence[index]
        x_begin = self.item2offset[index]
        x_end = x_begin + self.seq_len
        y_begin = x_end
        y_end = y_begin + self.pred_len
        prompt= self.prompt[sequence_index]
        
        seq = self.data[sequence_index][x_begin:y_end]
        # seq = np.concatenate((prompt, self.data[sequence_index][x_begin:y_end]))
        seq = torch.from_numpy(np.expand_dims(seq,0))
        seq_token, _, seq_scale = self.tokenizer.input_transform(seq)

        propmt_seq = torch.from_numpy(np.expand_dims(prompt,0))
        propmt_token, _, _ = self.tokenizer.input_transform(propmt_seq)

        seq_x = seq_token[0,:self.seq_len]
        seq_x = np.concatenate((propmt_token.squeeze(), seq_x), axis=0)
        data_y = np.concatenate((seq_scale, seq_token[0, self.seq_len:self.seq_len+self.pred_len]), axis=0)

        return seq_x, data_y

    def __len__(self):
        return self.num_items