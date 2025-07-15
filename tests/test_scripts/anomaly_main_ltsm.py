from ltsm.data_pipeline import AnomalyTrainingPipeline, anomaly_get_args, anomaly_seed_all

if __name__ == "__main__":
    args = anomaly_get_args()
    anomaly_seed_all(args.seed)
    pipeline = AnomalyTrainingPipeline(args)
    pipeline.run()