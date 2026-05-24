"""Rule-based department mapping config for MIMIC-IV mortality consultation.

This module is intentionally lightweight and standalone.
"""

DEPARTMENT_CANDIDATES = [
    "critical_care_hemodynamic",
    "respiratory_oxygenation",
    "metabolic_general",
]

# agent_1 is fixed by requirement.
FIXED_AGENT_ASSIGNMENT = {
    "agent_1": "critical_care_hemodynamic",
}

# Dynamic candidates for agent_2 and agent_3.
DYNAMIC_DEPARTMENTS = [
    "respiratory_oxygenation",
    "metabolic_general",
]

# Exact processed MIMIC-IV feature names from ehr_datasets/mimic-iv preprocessing.
MIMIC_IV_PROCESSED_FEATURES = {
    "Diastolic blood pressure",
    "Fraction inspired oxygen",
    "Glucose",
    "Heart Rate",
    "Height",
    "Mean blood pressure",
    "Oxygen saturation",
    "Respiratory rate",
    "Systolic blood pressure",
    "Temperature",
    "Weight",
    "pH",
}

# Exact feature-to-department map for deterministic assignment.
FEATURE_TO_DEPARTMENT = {
    "Diastolic blood pressure": "critical_care_hemodynamic",
    "Heart Rate": "critical_care_hemodynamic",
    "Mean blood pressure": "critical_care_hemodynamic",
    "Systolic blood pressure": "critical_care_hemodynamic",
    "Oxygen saturation": "respiratory_oxygenation",
    "Respiratory rate": "respiratory_oxygenation",
    "Fraction inspired oxygen": "respiratory_oxygenation",
    "Temperature": "respiratory_oxygenation",
    "pH": "respiratory_oxygenation",
    "Glucose": "metabolic_general",
    "Weight": "metabolic_general",
    "Height": "metabolic_general",
}

# Keyword lexicon for rule-based matching.
# Only lower-case matching is used.
DEPARTMENT_KEYWORDS = {
    "critical_care_hemodynamic": {
        "icu",
        "heart rate",
        "blood pressure",
        "mean blood pressure",
        "systolic blood pressure",
        "diastolic blood pressure",
    },
    "respiratory_oxygenation": {
        "oxygen saturation",
        "respiratory rate",
        "fraction inspired oxygen",
        "temperature",
        "ph",
    },
    "metabolic_general": {
        "glucose",
        "weight",
        "height",
        "critical",
    },
}

# Weights for different feature buckets.
BUCKET_WEIGHTS = {
    "abnormal_numeric": 1.4,
    "important_features": 0.9,
    "context": 0.7,
}
