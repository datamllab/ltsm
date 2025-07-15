from ltsm.data_pipeline import TokenizerTrainingPipeline, tokenizer_get_args, tokenizer_seed_all

if __name__ == "__main__":
    config = tokenizer_get_args()
    tokenizer_seed_all(config.seed)
    pipeline = TokenizerTrainingPipeline(config)
    pipeline.run()