import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

from utils.hparams import config as all_configs


META_KEYS = {"total_prompt_tokens", "total_completion_tokens", "total_time"}
SEED = 42
VAL_HOLDOUT = 0.15


def experiment_key(cfg: Dict):
    return (
        cfg["ehr_dataset_name"],
        cfg["ehr_task"],
        tuple(cfg["ehr_model_names"]),
        cfg["llm_name"],
        cfg["corpus_name"],
    )


def unique_configs(configs: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for cfg in configs:
        key = experiment_key(cfg)
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    return out


def fusion_dir_from_config(cfg: Dict) -> str:
    model_tag = "-".join(cfg["ehr_model_names"])
    llm = cfg["llm_name"]
    corpus = cfg["corpus_name"]
    task = cfg["ehr_task"]
    dataset = cfg["ehr_dataset_name"]
    return f"ehr_datasets/{dataset}/processed/fold_1/fusion_{task}_{model_tag}_{llm}_{corpus}"


def response_path_from_config(cfg: Dict, split: str) -> str:
    model_tag = "_".join(cfg["ehr_model_names"])
    return (
        f"response/{cfg['ehr_dataset_name']}_{cfg['ehr_task']}_{split}_"
        f"{model_tag}_{cfg['llm_name']}_{cfg['corpus_name']}/result.json"
    )


def load_split_arrays(cfg: Dict, split: str) -> Tuple[List, List[int], List[List[np.ndarray]]]:
    ehr_dir = cfg["ehr_dataset_dir"]
    pids = pd.read_pickle(f"{ehr_dir}/{split}_pid.pkl")
    y_raw = pd.read_pickle(f"{ehr_dir}/{split}_y.pkl")

    if cfg["ehr_task"] == "outcome":
        labels = [int(y[-1][0]) for y in y_raw]
    else:
        labels = [int(y[-1][2]) for y in y_raw]

    model_names = cfg["ehr_model_names"]
    ehr_embeddings_by_model = [
        pd.read_pickle(f"{ehr_dir}/dl_data/{name}_{cfg['ehr_task']}_{split}_embeddings.pkl")
        for name in model_names
    ]
    per_patient_ehr = [list(x) for x in zip(*ehr_embeddings_by_model)]
    return list(pids), labels, per_patient_ehr


def load_reports(report_json_path: str) -> Dict[str, str]:
    if not os.path.exists(report_json_path):
        raise FileNotFoundError(
            f"Missing report file: {report_json_path}. "
            "Run collaboration_pipeline.py for this experiment and split first."
        )

    raw = json.load(open(report_json_path, "r"))
    reports: Dict[str, str] = {}
    for key, value in raw.items():
        if str(key) in META_KEYS:
            continue
        if isinstance(value, dict):
            text = value.get("report", value.get("final_report", ""))
            if isinstance(text, str) and text.strip():
                reports[str(key)] = text
    return reports


def encode_reports(model, tokenizer, reports: Dict[str, str], device: torch.device) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for pid, text in reports.items():
        tokens = tokenizer(
            text,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)
        with torch.no_grad():
            hs = model(**tokens).last_hidden_state[0, 0, :].detach().cpu().numpy().astype(np.float32)
        out[pid] = hs
    return out


def build_patients_data(cfg: Dict, split: str, model, tokenizer, device: torch.device):
    report_path = response_path_from_config(cfg, split)
    reports = load_reports(report_path)

    pids, labels, per_patient_ehr = load_split_arrays(cfg, split)
    text_embeddings = encode_reports(model, tokenizer, reports, device)

    patients_data = {}
    for pid, y, ehr_emb in zip(pids, labels, per_patient_ehr):
        pid_s = str(pid)
        if pid_s not in reports:
            continue
        if pid_s not in text_embeddings:
            continue
        patients_data[pid_s] = {
            "label": y,
            "ehr_embeddings": ehr_emb,
            "text_embedding": text_embeddings[pid_s],
        }
    return patients_data, len(pids), len(reports)


def save_split(base_dir: str, split: str, patient_ids: List[str], patients_data: Dict[str, Dict]):
    text = [patients_data[pid]["text_embedding"] for pid in patient_ids]
    ehr = [patients_data[pid]["ehr_embeddings"] for pid in patient_ids]
    y = [patients_data[pid]["label"] for pid in patient_ids]

    pd.to_pickle(text, f"{base_dir}/{split}_text.pkl")
    pd.to_pickle(ehr, f"{base_dir}/{split}_ehr.pkl")
    pd.to_pickle(y, f"{base_dir}/{split}_y.pkl")
    pd.to_pickle(np.array(patient_ids), f"{base_dir}/{split}_pids.pkl")


def process_config(cfg: Dict, model, tokenizer, device: torch.device):
    out_dir = fusion_dir_from_config(cfg)
    os.makedirs(out_dir, exist_ok=True)

    test_data, test_total, test_reports = build_patients_data(cfg, "test", model, tokenizer, device)
    test_ids = list(test_data.keys())
    save_split(out_dir, "test", test_ids, test_data)

    val_data, val_total, val_reports = build_patients_data(cfg, "val", model, tokenizer, device)
    val_ids = np.array(list(val_data.keys()))
    val_labels = np.array([val_data[pid]["label"] for pid in val_ids])

    train_ids, holdout_ids = train_test_split(
        val_ids,
        test_size=VAL_HOLDOUT,
        random_state=SEED,
        stratify=val_labels,
    )
    save_split(out_dir, "train", list(train_ids), val_data)
    save_split(out_dir, "val", list(holdout_ids), val_data)

    print(
        f"[{cfg['ehr_dataset_name']}|{cfg['ehr_task']}|{'-'.join(cfg['ehr_model_names'])}] "
        f"test {len(test_ids)}/{test_total} (reports={test_reports}) | "
        f"train {len(train_ids)} val {len(holdout_ids)} from official val {val_total} (reports={val_reports})\\n"
        f"saved -> {out_dir}"
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("UFNLP/gatortron-base").to(device)
    tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-base")

    for cfg in unique_configs(all_configs):
        process_config(cfg, model, tokenizer, device)


if __name__ == "__main__":
    main()
