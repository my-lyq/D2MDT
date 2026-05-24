import os
from pathlib import Path

import pandas as pd
import shap
import torch
import numpy as np

from ehrdatasets.loader.datamodule import EhrDataModule
from ehrdatasets.loader.load_los_info import get_los_info
from pipelines import DlPipeline, MlPipeline
from configs.hparams import hparams, hparams_v2

shap.initjs()
torch.set_grad_enabled(True)
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "ehr_datasets"
BATCH_SIZE = 2048

def get_background_and_shap_variables(config):
    data_path = DATA_ROOT / f'{config["dataset"]}/processed/fold_{config["fold"]}'
    dm = EhrDataModule(str(data_path), batch_size=BATCH_SIZE, test_mode=config["mode"])
    x_bg, y_bg, lens_bg, pid_bg = (next(iter(dm.train_dataloader())))
    # only use the last visit to calculate the feature importance, and the shape is (batch_size, input_dim)
    x_bg = x_bg[:, -1, :].detach().cpu().numpy()
    x_shap, y_shap, lens_shap, pid_shap = (next(iter(dm.test_dataloader())))
    x_shap = x_shap[:, -1, :].detach().cpu().numpy()
    return x_bg, x_shap


def get_checkpoint_filename(ckpt_dir: str):
    filenames = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if len(filenames) == 1:
        filename = filenames[0]
    else:
        filenames.remove("best.ckpt") if "best.ckpt" in filenames else None
        filename = list(sorted(filenames, key=lambda x: int(x.split('-')[-1].split('.')[0][1:])))[-1]
    return filename


def get_feature_importance(config, x_bg, x_shap):
    # los_config = get_los_info(f'ehrdatasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    # config.update({"los_info": los_config})

    # checkpoint
    ckpt_dir = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}/checkpoints'
    filename = get_checkpoint_filename(ckpt_dir)
    ckpt_path = f'{ckpt_dir}/{filename}'

    # Keep model shape-related args aligned with training-time config in checkpoint.
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    ckpt_config = ckpt_obj.get("hyper_parameters", {}).get("config", {})
    for key in ["demo_dim", "lab_dim", "input_dim", "hidden_dim", "output_dim"]:
        if key in ckpt_config:
            config[key] = ckpt_config[key]

    if config["model"] in ["RF", "DT", "GBDT", "XGBoost", "CatBoost", "LR"]:
        pipeline = MlPipeline.load_from_checkpoint(ckpt_path, config=config)
    else:
        map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
        pipeline = DlPipeline.load_from_checkpoint(ckpt_path, map_location=map_location, config=config)

    def predict(x):
        output = pipeline.predict(x)
        return output

    # SHAP kmeans requires k <= number of distinct samples.
    unique_bg = np.unique(x_bg, axis=0).shape[0]
    if config["dataset"] == 'esrd':
        k = min(2, max(1, unique_bg))
    else:
        k = min(32, max(1, unique_bg))
    x_bg_kmeans = shap.kmeans(x_bg, k=k).data
    e = shap.KernelExplainer(predict, x_bg_kmeans)
    shap_values = e.shap_values(x_shap).squeeze()
    return shap_values

if __name__ == "__main__":
    hparam_set = os.environ.get("PYEHR_HPARAM_SET", "v1").lower()
    if hparam_set == "v2":
        best_hparams = hparams_v2
    else:
        best_hparams = hparams

    print(f"Using hparam set: {hparam_set} ({len(best_hparams)} models)")
    for i in range(len(best_hparams)):
        config = best_hparams[i]
        config["fold"] = 1
        config["seed"] = 0
        save_dir = DATA_ROOT / f"{config['dataset']}/processed/fold_1"
        os.makedirs(os.path.join(save_dir, 'dl_data'), exist_ok=True)

        for mode in ['val', 'test']:
            config["mode"] = mode
            print(config)
            x_bg, x_shap = get_background_and_shap_variables(config)
            shap_values = get_feature_importance(config, x_bg, x_shap)
            pd.to_pickle(shap_values, f"{save_dir}/dl_data/{config['model']}_{config['task']}_{mode}_shap.pkl")
            # shap_values = pd.read_pickle(f"{save_dir}/{config['model']}_{mode}_shap.pkl")

            if config["dataset"] in ['mimic-iv', 'mimic-iii']:
                feature_names_url = f'{save_dir}/labtest_features.pkl'
            else:
                feature_names_url = f'{save_dir}/numerical_features.pkl'
            test_raw_x_url = f'{save_dir}/{mode}_raw_x.pkl'
            feature_names = pd.read_pickle(feature_names_url)
            test_raw_x = pd.read_pickle(test_raw_x_url)

            # outs = pd.read_pickle(f"logs/test/{config['dataset']}/{config['task']}/{config['model']}/fold_1-seed_0/{mode}_outs.pkl")
            # shap_values = outs['feature_weight']

            if config["dataset"] in ['mimic-iv', 'mimic-iii']:
                feature_count = len(feature_names)
                features = shap_values[:, -feature_count:]
                test_raw_x = np.array([x[-1] for x in test_raw_x])[:, -feature_count:]
            else:
                features = shap_values[:, config["demo_dim"]:]
                test_raw_x = np.array([x[-1] for x in test_raw_x])[:, config["demo_dim"]:]
            print(features.shape, len(feature_names), test_raw_x.shape)

            all_features = []
            for feature_weight_item, raw_item in zip(features, test_raw_x):
                last_feat_dict = {key: {'value': value, 'attention': abs(attn)} for key, value, attn in zip(feature_names, raw_item, feature_weight_item)}
                last_feat_dict_sort = dict(sorted(last_feat_dict.items(), key=lambda x: abs(x[1]['attention']), reverse=True))
                selected_features = [item for item in last_feat_dict_sort.items()][:3]
                all_features.append(selected_features)
            pd.to_pickle(all_features, f"{save_dir}/dl_data/{config['model']}_{config['task']}_{mode}_features.pkl")
            
            outs = pd.read_pickle(f"logs/test/{config['dataset']}/{config['task']}/{config['model']}/fold_1-seed_0/{mode}_outs.pkl")
            preds = outs['preds'].tolist()
            pd.to_pickle(preds, f"{save_dir}/dl_data/{config['model']}_{config['task']}_{mode}_output.pkl")
            embeddings = outs['embeddings']
            pd.to_pickle(embeddings, f"{save_dir}/dl_data/{config['model']}_{config['task']}_{mode}_embeddings.pkl")