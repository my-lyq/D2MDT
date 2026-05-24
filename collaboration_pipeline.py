import os
from dotenv import load_dotenv
load_dotenv()

import json
import os
import traceback as tb
import time
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

# from utils.hparams import mimic_config as config
from utils.hparams import config as all_configs
from utils.framework import LeaderAgent, DoctorAgent, Collaboration
from utils.retrieve_utils import RetrievalSystem
from utils.healthcare_context_utils import get_data_from_files, ContextBuilder
from utils.department_assignment import build_assignment_signals, assign_departments
from utils.runner_utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(config)

leader_agent = LeaderAgent(llm_name=config["llm_name"], config=config)

print("load leader agent over")

retrieval_system = RetrievalSystem(retriever_name=config["retriever_name"], corpus_name=config["corpus_name"])

doctor_agents = [DoctorAgent(config, i, retrieval_system) for i in range(config["doctor_num"])]

print("load doctor agents over")

result_json = {}
all_prompt_tokens = 0
all_completion_tokens = 0
total_start = time.time()

test_pids, test_y = load_data(config)
preds = load_preds(config)

save_dir_base = f"{config['ehr_dataset_name']}_{config['ehr_task']}_{config['mode']}_{'_'.join(config['ehr_model_names'])}_{config['llm_name']}_{config['corpus_name']}"
config_name = str(config.get("config_name", "")).strip()
if config_name:
    save_dir = f"./response/{save_dir_base}_{config_name}"
else:
    save_dir = f"./response/{save_dir_base}"
os.makedirs(save_dir, exist_ok=True)
result_path = os.path.join(save_dir, "result.json")

if os.path.exists(result_path):
    try:
        result_json = json.load(open(result_path, "r"))
        all_prompt_tokens = int(result_json.get("total_prompt_tokens", 0))
        all_completion_tokens = int(result_json.get("total_completion_tokens", 0))
    except Exception:
        result_json = {}
        all_prompt_tokens = 0
        all_completion_tokens = 0


def flush_result_snapshot():
    result_json["total_prompt_tokens"] = all_prompt_tokens
    result_json["total_completion_tokens"] = all_completion_tokens
    result_json["total_time"] = f"{time.time() - total_start:.2f} s"
    with open(result_path, "w") as f:
        json.dump(result_json, f)


def _to_float_or_none(value):
    try:
        return float(value)
    except Exception:
        return None


def _minpse(preds, labels):
    precision, recall, _ = precision_recall_curve(labels, preds)
    return float(np.max([min(p, r) for p, r in zip(precision, recall)]))


def compute_prediction_metrics_from_pid_logs(root_dir):
    scores = []
    labels = []
    for name in os.listdir(root_dir):
        if not name.startswith("pid"):
            continue
        path = os.path.join(root_dir, name, "leader_logits_json.json")
        if not os.path.exists(path):
            continue
        try:
            payload = json.load(open(path, "r"))
        except Exception:
            continue

        score = _to_float_or_none(payload.get("final_logit"))
        label = _to_float_or_none(payload.get("ground_truth"))
        if score is None or label is None:
            continue
        scores.append(score)
        labels.append(int(label))

    if len(scores) == 0:
        return {
            "AUPRC": None,
            "AUROC": None,
            "min(+P, Se)": None,
            "num_eval_samples": 0,
        }

    y_score = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(labels, dtype=np.int64)

    auprc = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None
    auroc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None
    minpse = _minpse(y_score, y_true)

    return {
        "AUPRC": auprc,
        "AUROC": auroc,
        "min(+P, Se)": minpse,
        "num_eval_samples": int(len(y_true)),
    }


def compute_efficiency_metrics_from_pid_logs(root_dir):
    rows = []
    for name in os.listdir(root_dir):
        if not name.startswith("pid"):
            continue
        path = os.path.join(root_dir, name, "discussion_efficiency_summary.json")
        if not os.path.exists(path):
            continue
        try:
            rows.append(json.load(open(path, "r")))
        except Exception:
            continue

    def _avg(key):
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        vals = [float(v) for v in vals if _to_float_or_none(v) is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    return {
        "avg_rounds_executed": _avg("num_rounds_executed"),
        "avg_round1_prompt_token": _avg("round1_avg_prompt_token"),
        "avg_round2_prompt_token": _avg("round2_avg_prompt_token"),
        "avg_round1_user_prompt_length_chars": _avg("round1_avg_user_prompt_length_chars"),
        "avg_round2_user_prompt_length_chars": _avg("round2_avg_user_prompt_length_chars"),
        "avg_round1_non_retrieval_payload_length_chars": _avg("round1_avg_non_retrieval_payload_length_chars"),
        "avg_round2_non_retrieval_payload_length_chars": _avg("round2_avg_non_retrieval_payload_length_chars"),
        "num_efficiency_samples": int(len(rows)),
    }


def write_experiment_summary(cfg, root_dir):
    split = cfg.get("mode", "unknown")
    config_name = str(cfg.get("config_name", "unnamed_config"))
    summary = {
        "config_name": f"{config_name}_{split}",
        "split": split,
        "prediction_metrics": compute_prediction_metrics_from_pid_logs(root_dir),
        "efficiency_metrics": compute_efficiency_metrics_from_pid_logs(root_dir),
    }

    # Per-experiment summary.
    with open(os.path.join(root_dir, "experiment_results_summary.json"), "w") as f:
        json.dump(summary, f)

    # Global summary (all experiments) under response/.
    global_json_path = os.path.join("./response", "experiment_results_summary.json")
    existing = []
    if os.path.exists(global_json_path):
        try:
            existing = json.load(open(global_json_path, "r"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

    updated = [item for item in existing if item.get("config_name") != summary["config_name"]]
    updated.append(summary)
    with open(global_json_path, "w") as f:
        json.dump(updated, f)

    # Readable CSV table for paper-friendly comparison.
    csv_path = os.path.join("./response", "experiment_results_summary.csv")
    fieldnames = [
        "config_name",
        "split",
        "AUPRC",
        "AUROC",
        "min(+P, Se)",
        "num_eval_samples",
        "avg_rounds_executed",
        "avg_round1_prompt_token",
        "avg_round2_prompt_token",
        "avg_round1_user_prompt_length_chars",
        "avg_round2_user_prompt_length_chars",
        "avg_round1_non_retrieval_payload_length_chars",
        "avg_round2_non_retrieval_payload_length_chars",
        "num_efficiency_samples",
    ]

    rows = []
    for item in updated:
        pm = item.get("prediction_metrics", {})
        em = item.get("efficiency_metrics", {})
        rows.append(
            {
                "config_name": item.get("config_name"),
                "split": item.get("split"),
                "AUPRC": pm.get("AUPRC"),
                "AUROC": pm.get("AUROC"),
                "min(+P, Se)": pm.get("min(+P, Se)"),
                "num_eval_samples": pm.get("num_eval_samples"),
                "avg_rounds_executed": em.get("avg_rounds_executed"),
                "avg_round1_prompt_token": em.get("avg_round1_prompt_token"),
                "avg_round2_prompt_token": em.get("avg_round2_prompt_token"),
                "avg_round1_user_prompt_length_chars": em.get("avg_round1_user_prompt_length_chars"),
                "avg_round2_user_prompt_length_chars": em.get("avg_round2_user_prompt_length_chars"),
                "avg_round1_non_retrieval_payload_length_chars": em.get("avg_round1_non_retrieval_payload_length_chars"),
                "avg_round2_non_retrieval_payload_length_chars": em.get("avg_round2_non_retrieval_payload_length_chars"),
                "num_efficiency_samples": em.get("num_efficiency_samples"),
            }
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_patient_department_assignment(cfg, patient_index, patient_id):
    selected_payload = None
    selected_model = cfg["ehr_model_names"][0]
    for model_name in cfg["ehr_model_names"]:
        payload = get_data_from_files(
            cfg["ehr_dataset_dir"],
            cfg["ehr_task"],
            model_name,
            cfg["mode"],
            patient_index,
        )
        selected_payload = payload
        important_features = payload[-1]
        if important_features is not None and len(important_features) > 0:
            selected_model = model_name
            break

    if selected_payload is None:
        return {
            "agent_1": "critical_care_hemodynamic",
            "agent_2": "respiratory_oxygenation",
            "agent_3": "metabolic_general",
        }, {}, {"important_features_empty": True, "abnormal_numeric_features": []}, cfg["ehr_model_names"][0]

    _, raw_x, _, features, _, _, _, _, important_features = selected_payload

    context_builder = ContextBuilder(
        dataset_name=cfg["ehr_dataset_name"],
        task=cfg["ehr_task"],
        model_name=selected_model,
        mode=cfg["mode"],
    )
    _, _, hcontext = context_builder.generate_context(patient_index, patient_id)

    signals = build_assignment_signals(
        raw_x=raw_x,
        feature_names=features,
        hcontext=hcontext,
        important_features=important_features,
    )
    assignment, scores = assign_departments(
        abnormal_numeric_features=signals["abnormal_numeric_features"],
        context_text=signals["context_text"],
        important_feature_names=signals["important_feature_names"],
    )
    return assignment, scores, signals, selected_model

for patient_index, patient_id in tqdm(enumerate(test_pids), total=len(test_pids), desc=f"Processing patients in {config['ehr_dataset_name']} dataset {config['ehr_task']} task {config['mode']} mode"):
    start = time.time()
    sub_save_dir = f"{save_dir}/pid{patient_id}"
    if os.path.exists(os.path.join(sub_save_dir, "meta_final_summary.json")):
        if str(patient_id) not in result_json:
            try:
                payload = json.load(open(os.path.join(sub_save_dir, "meta_final_summary.json"), "r"))
                result_json[str(patient_id)] = {
                    "report": payload.get("final_report", ""),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "time": "0.00 s"
                }
                flush_result_snapshot()
            except Exception:
                pass
        continue

    os.makedirs(sub_save_dir, exist_ok=True)
    print('PatientID:', patient_id)
    analysis = []
    prompt_tokens = 0
    completion_tokens = 0
    basic_context = None

    try:
        assignment, assignment_scores, assignment_signals, assignment_model = compute_patient_department_assignment(config, patient_index, patient_id)

        for i, doctor_agent in enumerate(doctor_agents):
            if i == 0:
                assigned_department = assignment["agent_1"]
            elif i == 1:
                assigned_department = assignment["agent_2"]
            else:
                assigned_department = assignment["agent_3"]
            doctor_agent.set_assigned_department(assigned_department)
            doctor_agent.set_card_abnormal_numeric_features(assignment_signals.get("abnormal_numeric_features", []))

        with open(f"{sub_save_dir}/department_assignment.json", "w") as f:
            json.dump(
                {
                    "sample_id": str(patient_id),
                    "assignment_model": assignment_model,
                    "assignment": assignment,
                    "department_scores": assignment_scores,
                    "abnormal_numeric_features": assignment_signals.get("abnormal_numeric_features", []),
                    "important_features_empty": assignment_signals.get("important_features_empty", True),
                },
                f,
            )

        def process_agent(doctor_agent, patient_index, patient_id, i, sub_save_dir):
            response, basic_context, messages, prompt_token, completion_token = doctor_agent.analysis(patient_index, patient_id)

            with open(f"{sub_save_dir}/doctor{i + 1}_review.json", "w") as f:
                json.dump(response, f)

            with open(f"{sub_save_dir}/doctor{i + 1}_review_messages.json", "w") as f:
                json.dump(messages, f)

            with open(f"{sub_save_dir}/doctor{i + 1}_review_userprompt.txt", "w") as f:
                f.write(messages[1]["content"] if len(messages) > 1 else "")

            with open(f"{sub_save_dir}/doctor{i + 1}_analysis_prompt_log.json", "w") as f:
                json.dump(
                    {
                        "sample_id": str(patient_id),
                        "doctor_id": i + 1,
                        "model_name": doctor_agent.ehr_model_name,
                        "assigned_department": doctor_agent.assigned_department,
                        "original_query_preview": doctor_agent.last_original_query[:500],
                        "sanitized_query_preview": doctor_agent.last_sanitized_query[:500],
                        "query_sanitized_diff": doctor_agent.last_query_sanitized_diff,
                        "department_query_preview": doctor_agent.last_department_query[:500],
                        "retrieved_doc_count": doctor_agent.last_retrieved_doc_count,
                        "retrieved_doc_titles_preview": doctor_agent.last_retrieved_doc_titles,
                        "analysis_prompt_preview": (messages[1]["content"] if len(messages) > 1 else "")[:800],
                        "hcontext_with_department_preview": doctor_agent.last_hcontext_with_department[:800],
                    },
                    f,
                )

            with open(f"{sub_save_dir}/doctor{i + 1}_doctor_card_log.json", "w") as f:
                json.dump(
                    {
                        "doctor_card_raw": response.get("doctor_card_raw", ""),
                        "doctor_card_parsed": response.get("doctor_card_parsed", {}),
                        "doctor_card_parse_success": response.get("doctor_card_parse_success", False),
                        "doctor_card_compact_text_preview": str(response.get("doctor_card_compact_text", ""))[:800],
                        "semantic_card_enhanced": response.get("semantic_card_enhanced", False),
                        "mechanism_source": response.get("mechanism_source", "baseline"),
                        "support_feature_source": response.get("support_feature_source", "analysis_text"),
                    },
                    f,
                )

            return response, prompt_token, completion_token, basic_context

        with ThreadPoolExecutor(max_workers=config["doctor_num"]) as executor:
            futures = []
            for i, doctor_agent in enumerate(doctor_agents):
                futures.append(executor.submit(process_agent, doctor_agent, patient_index, patient_id, i, sub_save_dir))

            for future in futures:
                response, prompt_token, completion_token, bc = future.result()
                analysis.append(response)
                prompt_tokens += prompt_token
                completion_tokens += completion_token
                basic_context = bc

        # for i, doctor_agent in enumerate(doctor_agents):
        #     response, basic_context, messages, prompt_token, completion_token = doctor_agent.analysis(
        #         patient_index, patient_id)
        #     json.dump(response, open(
        #         f"{sub_save_dir}/doctor{i + 1}_review.json", "w"))
        #     analysis.append(response)
        #     prompt_tokens += prompt_token
        #     completion_tokens += completion_token
        #     json.dump(messages, open(
        #         f"{sub_save_dir}/doctor{i + 1}_review_messages.json", "w"))
        #     with open(f"{sub_save_dir}/doctor{i + 1}_review_userprompt.txt", "w") as f:
        #         f.write(messages[1]["content"])

        leader_agent.set_basic_info(basic_context)
        summary_content, messages, prompt_token, completion_token = leader_agent.summary(analysis, is_initial=True)
        json.dump(summary_content, open(f"{sub_save_dir}/meta_summary.json", "w"))
        json.dump(messages, open(f"{sub_save_dir}/meta_messages.json", "w"))
        with open(f"{sub_save_dir}/meta_userprompt.txt", "w") as f:
            f.write(messages[1]["content"])
        with open(f"{sub_save_dir}/leader_compact_report_log.json", "w") as f:
            json.dump(
                {
                    "leader_report_raw": summary_content.get("leader_report_raw", ""),
                    "leader_report_parsed": summary_content.get("leader_report_parsed", {}),
                    "leader_report_parse_success": summary_content.get("leader_report_parse_success", False),
                    "leader_report_compact_text_preview": str(summary_content.get("leader_report_compact_text", summary_content.get("report", "")))[:800],
                    "compact_leader_enabled": summary_content.get("compact_leader_enabled", False),
                    "report_source": summary_content.get("report_source", "fallback"),
                },
                f,
            )
        prompt_tokens += prompt_token
        completion_tokens += completion_token

        collaboration = Collaboration(leader_agent, doctor_agents, summary_content["answer"], summary_content["report"], sub_save_dir, doctor_num=config["doctor_num"], max_round=config["max_round"])
        current_report, _, prompt_token, completion_token = collaboration.collaborate()
        json.dump({"final_report": current_report}, open(f"{sub_save_dir}/meta_final_summary.json", "w"))
        prompt_tokens += prompt_token
        completion_tokens += completion_token

        logits, prompt_token, completion_token = leader_agent.revise_logits(analysis)
        if config["ehr_task"] == "outcome":
            logits['ground_truth'] = test_y[patient_index][-1][0]
        else:
            logits['ground_truth'] = test_y[patient_index][-1][2]
        json.dump(logits, open(f"{sub_save_dir}/leader_logits_json.json", "w"))
        prompt_tokens += prompt_token
        completion_tokens += completion_token

        end = time.time()
        result_json[patient_id] = {
            "report": current_report,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "time": f"{end - start:.2f} s"
        }

        all_prompt_tokens += prompt_tokens
        all_completion_tokens += completion_tokens
        flush_result_snapshot()
    except KeyboardInterrupt:
        flush_result_snapshot()
        raise
    except Exception as e:
        print(f"Error in patient {patient_id}")
        tb.print_exc()
        result_json[patient_id] = "Error"
        flush_result_snapshot()
        continue

flush_result_snapshot()

write_experiment_summary(config, save_dir)

processed = 0
for pid in test_pids:
    if os.path.exists(os.path.join(save_dir, f"pid{pid}", "meta_final_summary.json")):
        processed += 1
print(processed, "patients have been processed.")
