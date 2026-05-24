import os
from configs.hparams import hparams, hparams_v2, hparams_v3
import pandas as pd
import numpy as np

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt

from ehrdatasets.loader.datamodule import EhrDataModule
from ehrdatasets.loader.load_los_info import get_los_info
from metrics.bootstrap import run_bootstrap
from pipelines import DlPipeline, MlPipeline


def get_trainer_devices():
    device_idx = int(os.environ.get("PL_TRAINER_DEVICE", "0"))
    return [device_idx]


def get_processed_fold_dir(config):
    if "data_fold_dir" in config:
        return config["data_fold_dir"]

    obs_hours = int(config.get("observation_hours", int(os.environ.get("OBSERVATION_HOURS", "0"))))
    fold = config["fold"]
    if obs_hours > 0:
        return f"fold_{fold}_obs{obs_hours}h"
    return f"fold_{fold}"


def get_processed_data_path(config):
    return f'../ehr_datasets/{config["dataset"]}/processed/{get_processed_fold_dir(config)}'


def get_fold_tag(config):
    return get_processed_fold_dir(config).replace("fold_", "fold")


def get_checkpoint_filename(ckpt_dir: str):
    filenames = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if len(filenames) == 1:
        filename = filenames[0]
    else:
        filenames.remove("best.ckpt") if "best.ckpt" in filenames else None
        filename = list(sorted(filenames, key=lambda x: int(x.split('-')[-1].split('.')[0][1:])))[-1]
    return f"{ckpt_dir}/{filename}"


def align_config_dims_with_data(config):
    data_path = f'{get_processed_data_path(config)}/train_x.pkl'
    train_x = pd.read_pickle(data_path)
    if len(train_x) == 0:
        raise ValueError(f"Empty training set: {data_path}")

    feature_dim = int(np.asarray(train_x[0]).shape[-1])
    expected_input_dim = int(config["demo_dim"]) + int(config["lab_dim"])
    if expected_input_dim == feature_dim:
        return

    corrected_lab_dim = feature_dim - int(config["demo_dim"])
    if corrected_lab_dim <= 0:
        raise ValueError(
            f"Invalid config dims: feature_dim={feature_dim}, demo_dim={config['demo_dim']}, "
            f"lab_dim={config['lab_dim']}"
        )

    print(
        f"[DimFix] input_dim mismatch: config={expected_input_dim}, data={feature_dim}. "
        f"Auto-correct lab_dim: {config['lab_dim']} -> {corrected_lab_dim}"
    )
    config["lab_dim"] = corrected_lab_dim


def run_dl_experiment(config):
    # data
    dm = EhrDataModule(get_processed_data_path(config), batch_size=config["batch_size"])
    # logger
    checkpoint_filename = f'{config["model"]}-{get_fold_tag(config)}-seed{config["seed"]}'
    if "time_aware" in config and config["time_aware"] == True:
        checkpoint_filename+="-ta" # time-aware loss applied
    logger = CSVLogger(save_dir="logs", name=f'train/{config["dataset"]}/{config["task"]}', version=checkpoint_filename)

    # EarlyStop and checkpoint callback
    if config["task"] in ["outcome", "readmission", "multitask"]:
        # early_stopping_callback = EarlyStopping(monitor="auprc", patience=config["patience"], mode="max",)
        # checkpoint_callback = ModelCheckpoint(filename="best", monitor="auprc", mode="max")
        checkpoint_callback = ModelCheckpoint(every_n_epochs=50, filename="best")
    elif config["task"] == "los":
        early_stopping_callback = EarlyStopping(monitor="mae", patience=config["patience"], mode="min",)
        checkpoint_callback = ModelCheckpoint(filename="best", monitor="mae", mode="min")

    L.seed_everything(config["seed"]) # seed for reproducibility

    # train/val/test
    trainer = L.Trainer(accelerator="gpu", devices=get_trainer_devices(), max_epochs=config["epochs"], logger=logger, num_sanity_val_steps=0, callbacks=[checkpoint_callback])
    # trainer = L.Trainer(accelerator="gpu", devices=[1], max_epochs=1, logger=logger, num_sanity_val_steps=0, limit_train_batches=1, limit_val_batches=1, limit_test_batches=1)
    
    pipeline = DlPipeline(config)
    trainer.fit(pipeline, dm)
    # best_model_path = checkpoint_callback.best_model_path

    # best_model_path = get_checkpoint_filename(f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}/checkpoints')
    # pipeline = DlPipeline.load_from_checkpoint(best_model_path, config=config)
    # trainer.test(pipeline, dm)

    # perf = pipeline.test_performance
    # outs = pipeline.test_outputs
    # return perf, outs
    return None, None


def train_dl(config):
    # data
    dm = EhrDataModule(get_processed_data_path(config), batch_size=config["batch_size"])
    # logger
    checkpoint_filename = f'{config["model"]}-{get_fold_tag(config)}-seed{config["seed"]}'
    logger = CSVLogger(save_dir="logs", name=f'train/{config["dataset"]}/{config["task"]}', version=checkpoint_filename)

    # EarlyStop and checkpoint callback
    if config["task"] in ["outcome", "readmission", "multitask"]:
        early_stopping_callback = EarlyStopping(monitor="auprc", patience=config["patience"], mode="max",)
        checkpoint_callback = ModelCheckpoint(filename="best", monitor="auprc", mode="max")
    elif config["task"] == "los":
        early_stopping_callback = EarlyStopping(monitor="mae", patience=config["patience"], mode="min",)
        checkpoint_callback = ModelCheckpoint(filename="best", monitor="mae", mode="min")
    else:
        raise ValueError(f"Invalid task: {config['task']}")
    L.seed_everything(config["seed"]) # seed for reproducibility

    # train
    trainer = L.Trainer(accelerator="gpu", devices=get_trainer_devices(), max_epochs=config["epochs"], logger=logger, num_sanity_val_steps=0, callbacks=[checkpoint_callback, early_stopping_callback])
    pipeline = DlPipeline(config)
    trainer.fit(pipeline, dm)


def test_dl(config):
    # data
    dm = EhrDataModule(get_processed_data_path(config), batch_size=config["batch_size"], test_mode=config["mode"])
    
    # test
    trainer = L.Trainer(accelerator="gpu", devices=get_trainer_devices(), max_epochs=config["epochs"], logger=None, num_sanity_val_steps=0)
    best_model_path = get_checkpoint_filename(f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-{get_fold_tag(config)}-seed{config["seed"]}/checkpoints')
    pipeline = DlPipeline.load_from_checkpoint(best_model_path, config=config)
    trainer.test(pipeline, dm)
    
    perf = pipeline.test_performance
    outs = pipeline.test_outputs
    return perf, outs


def plot_distribution(data, save_dir, hidden_dim):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins='auto', edgecolor='black')
    ax.set_title('Distribution of Data', fontsize=16)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    # 保存图片
    plt.savefig(f'{save_dir}/{hidden_dim}_dis.png')


if __name__ == "__main__":
    hparam_set = os.environ.get("PYEHR_HPARAM_SET", "v1").lower()
    if hparam_set == "v2":
        best_hparams = hparams_v2
    elif hparam_set == "v3":
        best_hparams = hparams_v3
    else:
        best_hparams = hparams

    print(f"Using hparam set: {hparam_set} ({len(best_hparams)} models)")
    all_df = pd.DataFrame()
    for i in range(len(best_hparams)):
        config = best_hparams[i]
        config["fold"] = 1
        config["seed"] = 0
        if int(os.environ.get("OBSERVATION_HOURS", "0")) > 0:
            config["observation_hours"] = int(os.environ.get("OBSERVATION_HOURS", "0"))
        align_config_dims_with_data(config)
        best_metric = {}
        save_dir = f'logs/test/{config["dataset"]}/{config["task"]}/{config["model"]}/{get_processed_fold_dir(config)}-seed_{config["seed"]}'
        os.makedirs(save_dir, exist_ok=True)
        print(config)
        train_dl(config)
        for mode in ["val", "test"]:
            config["mode"] = mode
            perf, outs = test_dl(config)
            print(perf)
            # plot_distribution(outs['preds'], save_dir, config["hidden_dim"])
            metrics = run_bootstrap(outs['preds'], outs['labels'], config)
            metrics = {k: f"{v['mean']*100:.2f} ± {v['std']*100:.2f}" for k, v in metrics.items()}
            metrics_df = pd.DataFrame({'model': config["model"], 'mode': config["mode"], **metrics}, index=[i])
            print(metrics_df)
            pd.to_pickle(perf, f'{save_dir}/{mode}_perf.pkl')
            pd.to_pickle(outs, f'{save_dir}/{mode}_outs.pkl')
            all_df = pd.concat([all_df, metrics_df], axis=0)
        all_df.to_csv(f'{config["dataset"]}_{config["task"]}_metrics.csv', index=False)