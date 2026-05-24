mimic_iii_config = {
    "retriever_name": "MedCPT",
    "corpus_name": "MSD",
    "llm_name": "Qwen3-8B",
    # "llm_name": "GLM-4-Flash-250414",
    "epochs": 50,
    "patience": 10,
    "ehr_task": "outcome",
    "ehr_dataset_name": "mimic-iii",
    "ehr_dataset_dir": "./ehr_datasets/mimic-iii/processed/fold_1",
    "ehr_model_names": ["AdaCare", "MCGRU", "RETAIN"],
    "seeds": [0, 0, 0],
    "doctor_num": 3,
    "max_round": 3,
    "ehr_embed_dim": 128,
    "text_embed_dim": 1024,
    "merge_embed_dim": 128,
    "learning_rate": 1e-3,
    "main_metric": "auprc",
    "batch_size": 32,
    # "mode": "test",
    "mode": "val",
    "config_name": "DECA_full_mimic_iii",
}

mimic_iv_config = {
    "retriever_name": "MedCPT",
    "corpus_name": "MSD",
    # "llm_name": "GLM-4-Flash-250414",
    "llm_name": "Qwen3-8B",
    "epochs": 50,
    "patience": 10,
    "ehr_dataset_name": "mimic-iv",
    "ehr_task": "outcome",
    "ehr_dataset_dir": "./ehr_datasets/mimic-iv/processed/fold_1",
    "ehr_model_names": ["AdaCare", "MCGRU", "RETAIN"],
    "seeds": [0, 0, 0],
    "doctor_num": 3,
    "max_round": 2,
    "ehr_embed_dim": 128,
    "text_embed_dim": 1024,
    "merge_embed_dim": 128,
    "learning_rate": 1e-3,
    "main_metric": "auprc",
    "batch_size": 128,
    # "mode": "val",
    "mode": "test",
    "config_name": "DECA_full_mimic_iv",
}

mimic_config = dict(mimic_iii_config)
mimic_config_v2 = dict(mimic_iv_config)

config = [
    mimic_iv_config,
]
