# D2MDT: Department-aware MultiDisciplinary Team Consultation with Deliberation for Efficient Clinical Prediction

## Environmental Setups

- Create an environment `deptmdt` and activate it.

```bash
conda create -n deptmdt python=3.9
conda activate deptmdt
```

- Install the required packages.

```bash
pip install -r requirements.txt
```

## Usage

### Training EHR models in pyehr

- Run the following command to train the EHR models and obtain feature importance scores in `pyehr` directory.

```bash
cd pyehr
python train_test.py
python importance.py
```

> More details about the pre-processing and training steps can be found in the [`pyehr`](https://github.com/yhzhu99/pyehr) repository.

### Running LLM-based Multi-Agent Collaboration

Run the collaboration pipeline:

```bash
python collaboration_pipeline.py
```

- Write configurations in `utils/hparams.py`.
- Each experiment key is defined by:
  - `ehr_dataset_name`
  - `ehr_task`
  - `ehr_model_names`
  - `llm_name`
  - `corpus_name`

```python
# utils/hparams.py

mimic_config = {
    "retriever_name" : "MedCPT",
    "corpus_name" : "MSD",
    "llm_name" : "Qwen3-8b",
    "epochs" : 50,
    "patience" : 10,
    "ehr_dataset_name" : 'mimic-iii',
    "ehr_dataset_dir" : './ehr_datasets/mimic-iii/processed/fold_1',
    "ehr_model_names" : ['AdaCare', 'MCGRU', 'RETAIN'], 
    "seeds": [0, 0, 0],
    "doctor_num" : 3,
    "max_round" : 3,
    "ehr_embed_dim": 128,
    "text_embed_dim": 1024,
    "merge_embed_dim": 128,
    "learning_rate": 1e-3,
    "main_metric": "auprc",
    "batch_size": 32,
    "mode": "test"  
}
```

### Training Fusion Network

Run the following commands:

```bash
python utils/process_output.py
python train_fusion.py
```

## Datasets and Tasks

### Datasets

DeptMDT has been evaluated on the following datasets:

- MIMIC-III
- MIMIC-IV

### Tasks

DeptMDT has been evaluated on the following tasks:

- In-hospital Mortality Prediction
