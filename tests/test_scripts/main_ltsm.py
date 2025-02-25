from ltsm.data_pipeline import StatisticalTrainingPipeline, get_args, seed_all
import torch
import torch.nn as nn
import numpy as np

if __name__ == "__main__":
    config = get_args()
    seed = config.seed
    seed_all(seed)

    if config.model == "Informer":
        def collate_fn(batch):
            return {
                'input_data': torch.from_numpy(np.stack([x['input_data'] for x in batch])).type(torch.float32),
                'labels': torch.from_numpy(np.stack([x['labels'] for x in batch])).type(torch.float32),
                'timestamp_input': torch.from_numpy(np.stack([x['timestamp_input'] for x in batch])).type(torch.float32),
                'timestamp_labels': torch.from_numpy(np.stack([x['timestamp_labels'] for x in batch])).type(torch.float32)
            }
        
        def prediction_step(model, inputs, prediction_loss_only=False, ignore_keys=None):
            labels = inputs["labels"].to(model.module.device)
            input_data_mark = inputs["timestamp_input"].to(model.module.device)
            label_mark = inputs["timestamp_labels"].to(model.module.device)
            input_data = inputs["input_data"].to(model.module.device)
            
            outputs = model(input_data, input_data_mark, labels, label_mark)
            loss = nn.functional.mse_loss(outputs, labels)
            return (loss, outputs, labels)
        
        def compute_loss(model, inputs, return_outputs=False):
            input_data_mark = inputs["timestamp_input"].to(model.module.device)
            label_mark = inputs["timestamp_labels"].to(model.module.device)
            outputs = model(inputs["input_data"], input_data_mark, inputs["labels"], label_mark)
       
            loss = nn.functional.mse_loss(outputs, inputs["labels"])
            return (loss, outputs) if return_outputs else loss

        pipeline = StatisticalTrainingPipeline(config, 
                                               collate_fn=collate_fn, 
                                               prediction_step=prediction_step, 
                                               compute_loss=compute_loss)
    else:
        pipeline = StatisticalTrainingPipeline(config)

    pipeline.run()
