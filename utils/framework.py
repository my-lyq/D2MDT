from typing import Literal, List, Dict, Optional
import re
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from zai import ZhipuAiClient
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from utils.framework import *
from utils.retrieve_utils import RetrievalSystem
from utils.healthcare_context_utils import ContextBuilder
from utils.config import default_llm_name, deepseek_config, local_qwen_config, zhipu_config
from utils.prompt_template.template import *


class Agent:
    def __init__(
        self,
        role: Literal["doctor", "leader"],
        llm_name: str = default_llm_name
    ) -> None:
        self.role = role
        self.llm_name = llm_name

        # Set the llm_config according to your api_key and provider.
        if llm_name.lower() in {"qwen3-8b", "local-qwen3-8b", "local-qwen"}:
            self.llm_config = local_qwen_config
            self.client = OpenAI(
                api_key=self.llm_config["api_key"],
                base_url=self.llm_config["api_base"],
            )
        elif llm_name == "gpt-4o-mini":
            self.llm_config = deepseek_config
            self.client = OpenAI(api_key=self.llm_config["api_key"], base_url=self.llm_config["api_base"])
        elif "elsegpt" in llm_name.lower():
            self.llm_config = v1_config
            self.client = OpenAI(api_key=self.llm_config["api_key"], base_url=self.llm_config["api_base"])
        elif "qwen" in llm_name.lower() or "doubao" in llm_name.lower() or "claude" in llm_name.lower():
            self.llm_config = v2_config
            self.client = OpenAI(api_key=self.llm_config["api_key"], base_url=self.llm_config["api_base"])
        elif "llama" in llm_name.lower():
            self.llm_config = default_config
            self.client = OpenAI(api_key=self.llm_config["api_key"], base_url=self.llm_config["api_base"])
        elif "glm" in llm_name.lower() or "zhipu" in llm_name.lower():
            self.llm_config = zhipu_config
            self.client = ZhipuAiClient(api_key=self.llm_config["api_key"])


class DoctorAgent(Agent):
    def __init__(
        self,
        config: Dict,
        index: int = 0,
        retrieval_system: RetrievalSystem = None,
        role: str = "doctor",
    ) -> None:
        super().__init__(role, config["llm_name"])

        self.dataset_name = config['ehr_dataset_name']
        self.ehr_task = config['ehr_task']
        self.ehr_model_name = config['ehr_model_names'][index]
        self.mode = config['mode']
        self.assigned_department = None
        self.card_abnormal_numeric_features = []
        self.last_hcontext_with_department = ""
        self.last_department_query = ""
        self.last_retrieved_doc_count = 0
        self.last_retrieved_doc_titles = []
        self.current_round = 1
        self.delta_disagreement_points = "none"
        self.delta_leader_report_compact = ""
        self.latest_compact_text = ""
        self.retrieval_system = retrieval_system
        self.context_builder = ContextBuilder(dataset_name=self.dataset_name, task=self.ehr_task, model_name=self.ehr_model_name, mode=self.mode)

    @staticmethod
    def _infer_risk_vote(logit_value) -> str:
        value = None
        try:
            value = float(logit_value)
        except Exception:
            value = None

        if value is None:
            return "uncertain"
        if value >= 0.5:
            return "high"
        return "low"

    @staticmethod
    def _normalize_confidence(raw_conf) -> str:
        allowed = {"low", "medium", "high"}
        try:
            score = int(raw_conf)
            if score >= 3:
                return "high"
            if score <= 1:
                return "low"
            return "medium"
        except Exception:
            text = str(raw_conf).strip().lower()
            if text in allowed:
                return text
            return "medium"

    @staticmethod
    def _extract_support_features(text: str) -> List[str]:
        known_features = [
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
        ]
        lowered = text.lower()
        found = []
        for feat in known_features:
            if feat.lower() in lowered:
                found.append(feat)
        return found[:6]

    def _default_mechanisms_by_department(self) -> Dict[str, List[str]]:
        return {
            "critical_care_hemodynamic": [
                "hemodynamic instability",
                "perfusion compromise",
                "circulatory stress",
            ],
            "respiratory_oxygenation": [
                "oxygenation compromise",
                "respiratory stress",
                "hypoxic risk",
            ],
            "metabolic_general": [
                "metabolic stress",
                "physiologic instability",
                "acid-base imbalance",
            ],
        }

    def _semantic_mechanisms(self, analysis_text: str) -> tuple[List[str], str]:
        defaults = self._default_mechanisms_by_department().get(self.assigned_department, ["physiologic instability"])
        lowered = analysis_text.lower()

        detected = []
        if any(k in lowered for k in ["hemodynamic", "perfusion", "blood pressure", "heart rate", "circulatory"]):
            detected.extend(["hemodynamic instability", "perfusion compromise"])
        if any(k in lowered for k in ["oxygen", "respiratory", "fio2", "spo2", "hypox", "ph"]):
            detected.extend(["oxygenation compromise", "respiratory stress"])
        if any(k in lowered for k in ["glucose", "metabolic", "acid", "temperature", "instability"]):
            detected.extend(["metabolic stress", "acid-base imbalance"])

        dedup = []
        seen = set()
        for m in detected:
            if m in seen:
                continue
            seen.add(m)
            dedup.append(m)

        if dedup:
            merged = []
            for m in dedup + defaults:
                if m not in merged:
                    merged.append(m)
            return merged[:3], "mixed"
        return defaults[:3], "department_default"

    @staticmethod
    def _feature_status_phrase(feature_name: str) -> str:
        mapping = {
            "Mean blood pressure": "Mean blood pressure abnormal",
            "Systolic blood pressure": "Systolic blood pressure elevated",
            "Diastolic blood pressure": "Diastolic blood pressure abnormal",
            "Heart Rate": "Heart Rate elevated",
            "Oxygen saturation": "Oxygen saturation reduced",
            "Respiratory rate": "Respiratory rate elevated",
            "Fraction inspired oxygen": "FiO2 elevated",
            "Glucose": "Glucose abnormal",
            "Temperature": "Temperature abnormal",
            "pH": "pH abnormal",
            "Weight": "Weight abnormal",
            "Height": "Height baseline not informative",
        }
        return mapping.get(feature_name, f"{feature_name} abnormal")

    def _semantic_support_features(self, analysis_text: str) -> tuple[List[str], str]:
        support = []
        source = "analysis_text"

        if self.card_abnormal_numeric_features:
            source = "abnormal_numeric"
            for feat in self.card_abnormal_numeric_features:
                support.append(self._feature_status_phrase(str(feat)))

        if len(support) < 4:
            text_feats = self._extract_support_features(analysis_text)
            if text_feats:
                if source == "abnormal_numeric":
                    source = "mixed"
                for feat in text_feats:
                    phrase = self._feature_status_phrase(feat)
                    if phrase not in support:
                        support.append(phrase)

        if not support:
            source = "department_default"
            defaults = {
                "critical_care_hemodynamic": ["Mean blood pressure abnormal", "Heart Rate elevated"],
                "respiratory_oxygenation": ["Oxygen saturation reduced", "Respiratory rate elevated", "pH abnormal"],
                "metabolic_general": ["Glucose abnormal", "Temperature abnormal", "pH abnormal"],
            }
            support = defaults.get(self.assigned_department, ["physiologic instability noted"])

        return support[:5], source

    def _semantic_counter_evidence(self, support_phrases: List[str]) -> str:
        joined = " ".join(support_phrases).lower()
        if self.assigned_department == "respiratory_oxygenation":
            if "oxygen saturation reduced" not in joined and "respiratory rate elevated" not in joined:
                return "oxygenation not severely impaired"
        elif self.assigned_department == "metabolic_general":
            if "glucose abnormal" not in joined and "ph abnormal" not in joined:
                return "glucose not critically abnormal"
        elif self.assigned_department == "critical_care_hemodynamic":
            if "blood pressure" not in joined and "heart rate" not in joined:
                return "no strong hemodynamic deterioration"
        return "none"

    @staticmethod
    def _semantic_need_escalation(risk_vote: str, support_count: int, counter_evidence: str, mechanism_source: str) -> str:
        if support_count < 2:
            return "yes"
        if counter_evidence != "none":
            return "yes"
        if mechanism_source == "department_default" and support_count < 3:
            return "yes"
        if risk_vote in {"high", "low"} and support_count >= 3:
            return "no"
        return "yes"

    @staticmethod
    def _to_compact_text_from_card(parsed: Dict[str, str]) -> str:
        return (
            f"Risk vote: {parsed.get('risk_vote', 'uncertain')}. "
            f"Top mechanisms: {parsed.get('top_mechanisms', 'not specified')}. "
            f"Support features: {parsed.get('support_features', 'not specified')}. "
            f"Counter evidence: {parsed.get('counter_evidence', 'none')}. "
            f"Confidence: {parsed.get('confidence', 'medium')}. "
            f"Need escalation: {parsed.get('need_escalation', 'no')}."
        )

    @staticmethod
    def parse_doctor_card(card_raw: str) -> tuple[bool, Dict[str, str]]:
        required = [
            "risk_vote",
            "top_mechanisms",
            "support_features",
            "counter_evidence",
            "confidence",
            "need_escalation",
        ]
        parsed = {}
        for line in card_raw.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in required:
                parsed[key] = value

        success = all(k in parsed and str(parsed[k]).strip() for k in required)
        return success, parsed

    def _build_doctor_card(self, ans: Dict[str, str], analysis_text: str) -> str:
        # Step-5 baseline card builder (kept for fallback compatibility).
        risk_vote = self._infer_risk_vote(ans.get("logit"))
        support_features = self._extract_support_features(analysis_text)
        top_mechanisms = analysis_text.split(".")[0].strip() if analysis_text.strip() else "not specified"
        if not top_mechanisms:
            top_mechanisms = "not specified"

        confidence = self._normalize_confidence(ans.get("confidence"))

        lowered = analysis_text.lower()
        need_escalation = "yes" if ("uncertain" in lowered or "not sure" in lowered or confidence == "low") else "no"
        counter_evidence = "not explicitly stated"
        if "however" in lowered or "but" in lowered:
            counter_evidence = "contains conflicting indicators"

        lines = [
            f"risk_vote: {risk_vote}",
            f"top_mechanisms: {top_mechanisms}",
            f"support_features: {', '.join(support_features) if support_features else 'not specified'}",
            f"counter_evidence: {counter_evidence}",
            f"confidence: {confidence}",
            f"need_escalation: {need_escalation}",
        ]
        return "\n".join(lines)

    def _build_semantic_doctor_card(self, ans: Dict[str, str], analysis_text: str) -> tuple[str, str, str]:
        risk_vote = self._infer_risk_vote(ans.get("logit"))
        mechanisms, mechanism_source = self._semantic_mechanisms(analysis_text)
        support_phrases, support_source = self._semantic_support_features(analysis_text)
        counter_evidence = self._semantic_counter_evidence(support_phrases)
        confidence = self._normalize_confidence(ans.get("confidence"))
        need_escalation = self._semantic_need_escalation(
            risk_vote=risk_vote,
            support_count=len(support_phrases),
            counter_evidence=counter_evidence,
            mechanism_source=mechanism_source,
        )

        lines = [
            f"risk_vote: {risk_vote}",
            f"top_mechanisms: {' / '.join(mechanisms[:3])}",
            f"support_features: {'; '.join(support_phrases[:5])}",
            f"counter_evidence: {counter_evidence}",
            f"confidence: {confidence}",
            f"need_escalation: {need_escalation}",
        ]
        return "\n".join(lines), mechanism_source, support_source

    def set_assigned_department(self, department: Optional[str]) -> None:
        self.assigned_department = department

    def set_card_abnormal_numeric_features(self, abnormal_numeric_features: List[str]) -> None:
        self.card_abnormal_numeric_features = list(abnormal_numeric_features or [])

    def _department_hint(self, department: Optional[str]) -> str:
        if not department:
            return ""
        mapping = {
            "critical_care_hemodynamic": "Assigned department focus: critical care/hemodynamic instability, perfusion, blood pressure, heart rate",
            "respiratory_oxygenation": "Assigned department focus: respiratory/oxygenation, FiO2, SpO2, respiratory rate, pH",
            "metabolic_general": "Assigned department focus: metabolic/general instability, glucose, temperature, pH",
        }
        return mapping.get(department, "")

    def _department_retrieval_keywords(self, department: Optional[str]) -> str:
        if not department:
            return ""
        mapping = {
            "critical_care_hemodynamic": "hemodynamic instability, perfusion, blood pressure, heart rate",
            "respiratory_oxygenation": "oxygenation, FiO2, SpO2, respiratory rate, pH",
            "metabolic_general": "glucose, temperature, pH, general instability",
        }
        return mapping.get(department, "")

    @staticmethod
    def sanitize_retrieval_query(query: str) -> str:
        # Task-word normalization for MIMIC-IV ICU mortality retrieval.
        text = re.sub(r"\bESRD\s+patients\b", "ICU patients", query, flags=re.IGNORECASE)

        # Collapse repeated spaces/tabs while keeping line boundaries.
        normalized_lines = []
        for line in text.splitlines():
            line = re.sub(r"[ \t]+", " ", line).strip()
            normalized_lines.append(line)

        # Compress consecutive blank lines to a single blank line.
        compact_lines = []
        prev_blank = False
        for line in normalized_lines:
            is_blank = (line == "")
            if is_blank and prev_blank:
                continue
            compact_lines.append(line)
            prev_blank = is_blank

        return "\n".join(compact_lines).strip()

    def analysis(self, patient_index: int, patient_id: int) -> tuple[Dict[str, str], str, int, int]:
        basic_context, subcontext, healthcare_context = self.context_builder.generate_context(patient_index, patient_id)
        sanitized_query = self.sanitize_retrieval_query(subcontext)

        department_query = sanitized_query
        if self.assigned_department:
            keywords = self._department_retrieval_keywords(self.assigned_department)
            if keywords:
                department_query = sanitized_query + "\nDepartment retrieval focus: " + keywords

        self.last_department_query = department_query
        context = self.retrieve(department_query)
        hcontext_for_prompt = healthcare_context
        if self.assigned_department:
            hint = self._department_hint(self.assigned_department)
            if hint:
                hcontext_for_prompt = hint + "\n" + healthcare_context

        self.patient_info = hcontext_for_prompt
        self.last_hcontext_with_department = hcontext_for_prompt

        system_prompt = doctor_review_system
        user_prompt = doctor_review_user.render(context=context, hcontext=hcontext_for_prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        ans, prompt_token, completion_token = self.invoke(messages=messages)

        analysis_text = self._pick_text(ans, ["analysis", "reason", "report", "answer"])
        mechanism_source = "baseline"
        support_source = "analysis_text"
        try:
            card_text, mechanism_source, support_source = self._build_semantic_doctor_card(ans, analysis_text)
        except Exception:
            # Safety fallback: never break the main pipeline on card generation.
            card_text = self._build_doctor_card(ans, analysis_text)
            mechanism_source = "baseline_fallback"
            support_source = "analysis_text"

        card_ok, card_fields = self.parse_doctor_card(card_text)
        compact_text = self._to_compact_text_from_card(card_fields) if card_ok else analysis_text
        ans["doctor_card_compact_text"] = compact_text
        ans["mechanism_source"] = mechanism_source
        ans["support_feature_source"] = support_source
        ans["analysis"] = compact_text

        self.latest_analysis = analysis_text
        self.initial_analysis = analysis_text

        return ans, basic_context, messages, prompt_token, completion_token

    def collaboration(self, leader_opinion, leader_report) -> tuple[Dict[str, str], int, int]:
        """Doctor agent collaboration with leader agent.

        Returns:
            Dict[str, str | float]: {
                "answer": agree / disagree.
                "confidence": a float number between 0 and 1.
                "reason": doctor agent output part.
                "evidences": doctors' new rag documents.
            }
        """
        context = self.retrieve(self.latest_analysis)

        analysis_input = self.latest_analysis
        opinion_input = leader_opinion
        report_input = leader_report
        uses_delta_discussion = self.current_round >= 2
        used_compact_doctor_text = False
        used_compact_leader_report = False
        delta_fallback_used = False

        compact_doctor_text = ""
        compact_leader_report = ""
        disagreement_points = "none"

        if uses_delta_discussion:
            compact_doctor_text = str(self.latest_compact_text or "").strip()
            compact_leader_report = str(self.delta_leader_report_compact or "").strip()
            disagreement_points = str(self.delta_disagreement_points or "none").strip() or "none"

            if compact_doctor_text and compact_leader_report:
                analysis_input = compact_doctor_text
                opinion_input = disagreement_points
                report_input = compact_leader_report
                used_compact_doctor_text = True
                used_compact_leader_report = True
            else:
                delta_fallback_used = True

        system_prompt = doctor_collaboration_system
        user_prompt = doctor_collaboration_user.render(context=context, analysis=analysis_input, opinion=opinion_input, report=report_input)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        retrieval_context_length_chars = len(str(context))
        non_retrieval_payload_length_chars = len(str(analysis_input)) + len(str(opinion_input)) + len(str(report_input))
        collaboration_user_prompt_length_chars = len(str(user_prompt))

        ans, prompt_token, completion_token = self.invoke(messages=messages)

        reason_text = self._pick_text(ans, ["reason", "analysis", "report", "answer"])
        evidences = ans.get("evidences", [])
        if not isinstance(evidences, list):
            evidences = [str(evidences)]
        evidences = [str(item) for item in evidences]

        ans["reason"] = reason_text
        ans["evidences"] = evidences
        ans["used_compact_doctor_text"] = used_compact_doctor_text
        ans["used_compact_leader_report"] = used_compact_leader_report
        ans["delta_fallback_used"] = delta_fallback_used
        ans["disagreement_points"] = str(self.delta_disagreement_points or "none") if uses_delta_discussion else "none"
        ans["round_number"] = int(self.current_round)
        ans["retrieval_context_length_chars"] = retrieval_context_length_chars
        ans["non_retrieval_payload_length_chars"] = non_retrieval_payload_length_chars
        ans["collaboration_user_prompt_length_chars"] = collaboration_user_prompt_length_chars
        ans["collaboration_prompt_token"] = int(prompt_token)
        ans["collaboration_completion_token"] = int(completion_token)
        ans["compact_doctor_text_length_chars"] = len(compact_doctor_text)
        ans["compact_leader_report_length_chars"] = len(compact_leader_report)
        ans["disagreement_points_length_chars"] = len(disagreement_points)
        self.latest_analysis = reason_text + ("\n".join(evidences) if evidences else "")
        compact_reason = reason_text.split(".")[0].strip() if isinstance(reason_text, str) else ""
        self.latest_compact_text = compact_reason if compact_reason else reason_text
        return ans, messages, prompt_token, completion_token

    @staticmethod
    def _pick_text(payload: Dict, preferred_keys: List[str]) -> str:
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(payload, ensure_ascii=False)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def invoke(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                stream=False
            )
            content = response.choices[0].message.content
            prompt_token = response.usage.prompt_tokens
            completion_token = response.usage.completion_tokens
            ans = extract_and_parse_json(content)
        except Exception as e:
            raise e
        end = time.time()
        print(f"DoctorAgent time cost: {end - start:.2f}s.")
        return ans, prompt_token, completion_token

    def retrieve(self, question: str, k: int = 3) -> str:
        if self.retrieval_system is None:
            return "No retrieval system is available."
        retrieve_texts, _, _ = self.retrieval_system.retrieve(question, k=k)
        self.last_retrieved_doc_count = len(retrieve_texts)
        self.last_retrieved_doc_titles = [str(item.get("title", "")) for item in retrieve_texts[:5] if isinstance(item, dict)]
        contexts = [
            "Document [{:d}] (Title: {:s}) {:s}".format(
                idx + 1,
                retrieve_texts[idx].get("title", ""),
                retrieve_texts[idx].get("content", ""),
            )
            for idx in range(len(retrieve_texts))
        ]
        return "\n".join(contexts)
class LeaderAgent(Agent):
    def __init__(
        self,
        role: str = "leader",
        llm_name: str = default_llm_name,
        config: Dict = None,
    ) -> None:
        super().__init__(role, llm_name)
        config = config or {}
        self.action_system_prompt = action_system
        self.action_user_prompt = action_user
        self.current_report = ""
        self.basic_info = None

    @staticmethod
    def _risk_label_from_answer(answer_text: str) -> str:
        text = str(answer_text).lower()
        if "high" in text:
            return "high"
        if "low" in text:
            return "low"
        return "medium"

    @staticmethod
    def _normalize_consensus_label(score: float) -> str:
        if score >= 0.8:
            return "strong"
        if score >= 0.5:
            return "moderate"
        return "weak"

    def _consensus_from_doctors(self, doctor_responses: List[Dict], is_initial: bool) -> str:
        if not doctor_responses:
            return "weak"

        if is_initial:
            votes = []
            for res in doctor_responses:
                card = self._parse_compact_doctor_card_text(res.get("doctor_card_compact_text", ""))
                card_vote = str(card.get("risk_vote", "")).lower()
                if card_vote in {"high", "low"}:
                    votes.append(card_vote)
                    continue
                try:
                    logit = float(res.get("logit"))
                    votes.append("high" if logit >= 0.5 else "low")
                except Exception:
                    continue
            if not votes:
                return "weak"
            majority = max(votes.count("high"), votes.count("low")) / len(votes)
            return self._normalize_consensus_label(majority)

        agree_count = 0.0
        total_weight = 0.0
        for res in doctor_responses:
            opinion = str(res.get("answer", "")).lower()
            conf_raw = res.get("confidence", 1)
            try:
                conf = max(1, min(3, int(conf_raw)))
            except Exception:
                conf = 1
            total_weight += conf
            if "agree" in opinion:
                agree_count += conf

        if total_weight <= 0:
            return "weak"
        return self._normalize_consensus_label(max(agree_count / total_weight, 1.0 - agree_count / total_weight))

    @staticmethod
    def _parse_compact_doctor_card_text(card_text: str) -> Dict[str, str]:
        parsed = {}
        text = str(card_text or "").strip()
        if not text:
            return parsed

        normalized = text.replace("\n", ";")
        for seg in normalized.split("."):
            if ":" not in seg:
                continue
            key, value = seg.split(":", 1)
            normalized_key = key.strip().lower().replace(" ", "_")
            if normalized_key in {
                "risk_vote",
                "top_mechanisms",
                "support_features",
                "counter_evidence",
                "confidence",
                "need_escalation",
            }:
                parsed[normalized_key] = value.strip()
        return parsed

    @staticmethod
    def _split_mechanisms(text: str) -> List[str]:
        tokens = []
        for chunk in re.split(r"[;/,]|\s+/\s+", str(text)):
            t = chunk.strip()
            if t and t.lower() not in {"none", "not specified"}:
                tokens.append(t)
        return tokens

    def _collect_key_mechanisms(self, doctor_responses: List[Dict], is_initial: bool) -> tuple[List[str], str]:
        out = []
        source = "fallback"

        if is_initial:
            for res in doctor_responses:
                card = self._parse_compact_doctor_card_text(res.get("doctor_card_compact_text", ""))
                m = card.get("top_mechanisms", "") if isinstance(card, dict) else ""
                for token in self._split_mechanisms(m):
                    if token not in out:
                        out.append(token)
            if out:
                return out[:3], "doctor_cards"

        for res in doctor_responses:
            reason = str(res.get("reason") or res.get("analysis") or "")
            lowered = reason.lower()
            if any(k in lowered for k in ["hemodynamic", "blood pressure", "heart rate", "perfusion"]):
                if "hemodynamic instability" not in out:
                    out.append("hemodynamic instability")
            if any(k in lowered for k in ["oxygen", "respiratory", "spo2", "fio2", "hypox", "ph"]):
                if "oxygenation compromise" not in out:
                    out.append("oxygenation compromise")
            if any(k in lowered for k in ["glucose", "metabolic", "acid", "temperature"]):
                if "metabolic stress" not in out:
                    out.append("metabolic stress")

        if out:
            source = "mixed"
            return out[:3], source

        return ["physiologic instability"], source

    def _collect_support_summary(self, doctor_responses: List[Dict], is_initial: bool) -> List[str]:
        summary = []
        if is_initial:
            for res in doctor_responses:
                evidences = res.get("evidences", [])
                if isinstance(evidences, list):
                    for ev in evidences:
                        text = str(ev).strip()
                        if text and text not in summary:
                            summary.append(text[:120])
                            if len(summary) >= 4:
                                return summary
        for res in doctor_responses:
            reason = str(res.get("reason") or res.get("analysis") or "").strip()
            if reason:
                snippet = reason.split(".")[0].strip()
                if snippet and snippet not in summary:
                    summary.append(snippet[:120])
            if len(summary) >= 4:
                break
        return summary[:4]

    def _minority_and_discussion_points(self, doctor_responses: List[Dict], is_initial: bool) -> tuple[str, str]:
        minority = "none"
        points = []

        if is_initial:
            votes = []
            for res in doctor_responses:
                card = self._parse_compact_doctor_card_text(res.get("doctor_card_compact_text", ""))
                vote = str(card.get("risk_vote", "")).lower()
                if vote in {"high", "low"}:
                    votes.append(vote)
            if votes and len(set(votes)) > 1:
                minority = "minority diagnostic direction differs"
                points.append("reconcile risk disagreement across doctors")
        else:
            disagree_reasons = []
            for res in doctor_responses:
                opinion = str(res.get("answer", "")).lower()
                if "disagree" in opinion:
                    r = str(res.get("reason", "")).strip()
                    if r:
                        disagree_reasons.append(r.split(".")[0][:120])
            if disagree_reasons:
                minority = disagree_reasons[0]
                points.append("address minority disagreement evidence")

        if not points:
            points_text = "none"
        else:
            points_text = "; ".join(points[:2])
        return minority, points_text

    @staticmethod
    def parse_leader_compact_report(report_raw: str) -> tuple[bool, Dict[str, str]]:
        required = [
            "final_risk",
            "consensus_level",
            "key_mechanisms",
            "support_summary",
            "minority_view",
            "discussion_points",
        ]
        parsed = {}
        for line in str(report_raw).splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            key = k.strip().lower()
            if key in required:
                parsed[key] = v.strip()
        success = all(k in parsed and parsed[k] for k in required)
        return success, parsed

    @staticmethod
    def _leader_compact_text(parsed: Dict[str, str]) -> str:
        return (
            f"final_risk: {parsed.get('final_risk', 'medium')}; "
            f"consensus_level: {parsed.get('consensus_level', 'moderate')}; "
            f"key_mechanisms: {parsed.get('key_mechanisms', 'physiologic instability')}; "
            f"support_summary: {parsed.get('support_summary', 'none')}; "
            f"minority_view: {parsed.get('minority_view', 'none')}; "
            f"discussion_points: {parsed.get('discussion_points', 'none')}"
        )

    def _build_compact_leader_report(self, ans: Dict[str, str], doctor_responses: List[Dict], is_initial: bool) -> tuple[str, str]:
        final_risk = self._risk_label_from_answer(ans.get("answer", ""))
        consensus_level = self._consensus_from_doctors(doctor_responses, is_initial)
        mechanisms, mech_source = self._collect_key_mechanisms(doctor_responses, is_initial)
        support = self._collect_support_summary(doctor_responses, is_initial)
        minority_view, discussion_points = self._minority_and_discussion_points(doctor_responses, is_initial)

        lines = [
            f"final_risk: {final_risk}",
            f"consensus_level: {consensus_level}",
            f"key_mechanisms: {' / '.join(mechanisms[:3]) if mechanisms else 'physiologic instability'}",
            f"support_summary: {' | '.join(support[:4]) if support else 'none'}",
            f"minority_view: {minority_view}",
            f"discussion_points: {discussion_points}",
        ]

        source = "doctor_cards" if (is_initial and mech_source == "doctor_cards") else ("mixed" if mech_source != "fallback" else "fallback")
        return "\n".join(lines), source

    def set_basic_info(self, patient_info: str) -> None:
        self.basic_info = patient_info

    def generate_doctors_prompt(self, doctor_responses: List[Dict], is_initial=False) -> str:
        responses = ""
        for idx, res in enumerate(doctor_responses):
            if is_initial:
                if 'evidences' in res and res['evidences']:
                    evidence = res['evidences']
                    if not isinstance(evidence[0], str):
                        evidence = list(map(str, evidence))
                    evidence = '\n'.join(evidence)
                else:
                    evidence = ""
                analysis_payload = str(res.get('doctor_card_compact_text') or res.get('analysis') or "")
                response = initial_doctor_summary.render(diagnosis=res['logit'], analysis=analysis_payload, evidence=evidence)
            else:
                documents = '\n'.join(res['evidences']) if 'evidences' in res else ""
                reason = str(res.get('reason') or res.get('analysis') or res.get('raw') or "")
                opinion = str(res.get('answer') or "disagree")
                try:
                    confidence = int(res.get('confidence', 1))
                except Exception:
                    confidence = 1

                if confidence == 1:
                    response = doctor_low_confidence.render(reason=reason, opinion=opinion, documents=documents)
                elif confidence == 2:
                    response = doctor_moderate_confidence.render(reason=reason, opinion=opinion, documents=documents)
                elif confidence == 3:
                    response = doctor_high_confidence.render(reason=reason, opinion=opinion, documents=documents)
                else:
                    print(f"Doctor Confidence Error: {res.get('confidence')}!")
            response = "Doctor " + str(idx + 1) + "'s statement is as following:\n" + response + "\n"
            responses = responses + response
        return responses

    def extract_action_response(self, response: Dict):
        if "yes" in response['action'].lower():
            return 1
        elif "no" in response['action'].lower():
            return 0
        else:
            raise ValueError(f"Invalid action response: {response['action']}.")

    def next_action(self, doctor_responses) -> tuple[int, Dict[str, str], int, int]:
        """
        doctor_responses的输入形式:[
            {
                "answer": agree / disagree
                "confidence": 0 - 1
                "reason": doctor llm output part
                "evidences": doctors' new rag documents
            },
            ...
        ]
        输出: 1 代表继续循环; 0 代表结束循环
        """
        action_user_prompt = self.generate_doctors_prompt(doctor_responses)
        messages = [
            {"role": "system", "content": action_system},
            {"role": "user", "content": action_user.render(opinions=action_user_prompt)}
        ]
        ans, prompt_token, completion_token = self.invoke(messages)
        action = self.extract_action_response(ans)
        return action, ans, messages, prompt_token, completion_token

    def summary(self, doctor_responses, is_initial=False) -> tuple[Dict[str, str], str, int, int]:
        """ 
        依据：
        1. patient_info: 病人基本信息
        2. doctor_answer: 整理后的doctor_responses
        3. current_report: 上一轮的病历总结
        (agreed_evidence: 被认可的证据列表)
        """
        doctors_answer_prompt = self.generate_doctors_prompt(doctor_responses, is_initial)
        if is_initial:
            messages = [
                {"role": "system", "content": meta_summary_system},
                {"role": "user", "content": meta_summary_user.render(patient_info=self.basic_info, doctor_info=doctors_answer_prompt)}
            ]
            ans, prompt_token, completion_token = self.invoke(messages)
            report_source = "fallback"
            try:
                report_text, report_source = self._build_compact_leader_report(ans, doctor_responses, is_initial=True)
                report_ok, report_fields = self.parse_leader_compact_report(report_text)
                compact_text = self._leader_compact_text(report_fields) if report_ok else str(ans.get("report", ""))
            except Exception:
                compact_text = str(ans.get("report", ""))
                report_source = "fallback"

            ans["report_source"] = report_source
            ans["report"] = compact_text
            ans["leader_report_compact_text"] = compact_text
            self.current_report = ans['answer'] + "\n" + ans['report']
            return ans, messages, prompt_token, completion_token
        else:
            messages = [
                {"role": "system", "content": collaborative_summary_system},
                {"role": "user", "content": collaborative_summary_user.render(doctor_info=doctors_answer_prompt, latest_info=self.current_report)}
            ]
            ans, prompt_token, completion_token = self.invoke(messages)
            action = self.extract_action_response(ans)
            report_source = "fallback"
            try:
                report_text, report_source = self._build_compact_leader_report(ans, doctor_responses, is_initial=False)
                report_ok, report_fields = self.parse_leader_compact_report(report_text)
                compact_text = self._leader_compact_text(report_fields) if report_ok else str(ans.get("report", ""))
            except Exception:
                compact_text = str(ans.get("report", ""))
                report_source = "fallback"

            ans["report_source"] = report_source
            ans["report"] = compact_text
            ans["leader_report_compact_text"] = compact_text
            self.current_report = ans['answer'] + "\n" + ans['report']
            return action, ans, messages, prompt_token, completion_token

    def generate_doctors_initial_prompt(self, doctor_responses: List[Dict]) -> str:
        responses = ""
        for idx in range(len(doctor_responses)):
            i = doctor_responses[idx]
            prompt = initial_doctor_for_revise.render(diagnosis=i['logit'], analysis=i['analysis'])
            if idx < len(doctor_responses) - 1:
                response = "Doctor " + str(idx) + "'s statement is as following:\n" + prompt + "\n"
            else:
                response = "Doctor " + str(idx) + "'s statement is as following:\n" + prompt
            responses = responses + response
        return responses

    def revise_logits(self, doctor_initial_responses) -> tuple[Dict[str, str], int, int]:
        doctors_answer_prompt = self.generate_doctors_initial_prompt(doctor_initial_responses)
        messages = [
            {"role": "system", "content": revise_logits_system},
            {"role": "user", "content": revise_logits_user.render(doctor_info=doctors_answer_prompt, latest_info=self.current_report)}
        ]
        ans, prompt_token, completion_token = self.invoke(messages)
        return ans, prompt_token, completion_token

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def invoke(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                stream=False
            )
            content = response.choices[0].message.content
            prompt_token = response.usage.prompt_tokens
            completion_token = response.usage.completion_tokens
            ans = extract_and_parse_json(content)
        except Exception as e:
            raise e
        end = time.time()
        print(f"MetaAgent time cost: {end - start:.2f}s.")
        return ans, prompt_token, completion_token


class Collaboration():
    def __init__(self, leader_agent: LeaderAgent, doctor_agents: List[DoctorAgent], leader_opinion: str, leader_report: str, save_dir: str, doctor_num: int = 3, max_round: int = 3):
        self.leader_opinion = leader_opinion
        self.leader_report = leader_report
        self.doctor_responses = []
        self.doctor_num = doctor_num
        self.leader_agent = leader_agent
        self.doctor_agents = doctor_agents
        self.max_round = max_round
        self.now_round = 0
        self.save_dir = save_dir
        self.last_leader_report_fields = self._parse_compact_leader_report_text(leader_report)

    @staticmethod
    def _parse_compact_leader_report_text(report_text: str) -> Dict[str, str]:
        parsed = {}
        text = str(report_text or "").strip()
        if not text:
            return parsed

        normalized = text.replace("\n", ";")
        for seg in normalized.split(";"):
            if ":" not in seg:
                continue
            k, v = seg.split(":", 1)
            key = k.strip().lower()
            if key in {
                "final_risk",
                "consensus_level",
                "key_mechanisms",
                "support_summary",
                "minority_view",
                "discussion_points",
            }:
                parsed[key] = v.strip()
        return parsed

    @staticmethod
    def _parse_compact_doctor_card_text(card_text: str) -> Dict[str, str]:
        parsed = {}
        text = str(card_text or "").strip()
        if not text:
            return parsed

        normalized = text.replace("\n", ";")
        for seg in normalized.split("."):
            if ":" not in seg:
                continue
            key, value = seg.split(":", 1)
            normalized_key = key.strip().lower().replace(" ", "_")
            if normalized_key in {
                "risk_vote",
                "top_mechanisms",
                "support_features",
                "counter_evidence",
                "confidence",
                "need_escalation",
            }:
                parsed[normalized_key] = value.strip()
        return parsed

    @staticmethod
    def _is_none_text(value: str) -> bool:
        return str(value or "").strip().lower() in {"", "none", "not specified"}

    def _extract_disagreement_points(self, prev_responses: List[Dict]) -> str:
        points = str(self.last_leader_report_fields.get("discussion_points", "none"))
        if not self._is_none_text(points):
            return points

        minority = str(self.last_leader_report_fields.get("minority_view", "none"))
        if not self._is_none_text(minority):
            return minority

        risk_votes = set()
        escalation_yes = False
        mechanisms = set()

        for res in prev_responses or []:
            card = self._parse_compact_doctor_card_text(res.get("doctor_card_compact_text", ""))
            if isinstance(card, dict):
                rv = str(card.get("risk_vote", "")).strip().lower()
                if rv in {"high", "low", "uncertain"}:
                    risk_votes.add(rv)

                need = str(card.get("need_escalation", "")).strip().lower()
                if need == "yes":
                    escalation_yes = True

                top = str(card.get("top_mechanisms", "")).strip()
                if top and top.lower() != "not specified":
                    mechanisms.add(top.lower())

        derived = []
        if len(risk_votes) > 1:
            derived.append("risk_vote disagreement")
        if escalation_yes:
            derived.append("escalation requested")
        if len(mechanisms) > 1:
            derived.append("mechanism mismatch")

        return "; ".join(derived) if derived else "none"

    def _leader_report_for_delta(self) -> tuple[str, bool]:
        if self.last_leader_report_fields:
            compact_text = (
                f"final_risk: {self.last_leader_report_fields.get('final_risk', 'medium')}; "
                f"consensus_level: {self.last_leader_report_fields.get('consensus_level', 'moderate')}; "
                f"key_mechanisms: {self.last_leader_report_fields.get('key_mechanisms', 'physiologic instability')}; "
                f"support_summary: {self.last_leader_report_fields.get('support_summary', 'none')}; "
                f"minority_view: {self.last_leader_report_fields.get('minority_view', 'none')}; "
                f"discussion_points: {self.last_leader_report_fields.get('discussion_points', 'none')}"
            )
            return compact_text, True
        return self.leader_report, False

    def collaborate(self):
        assert len(self.doctor_responses) == 0 and self.now_round == 0
        prompt_tokens = 0
        completion_tokens = 0
        efficiency_records = []

        while self.now_round < self.max_round:
            prev_round_responses = list(self.doctor_responses)
            self.now_round += 1
            self.doctor_responses = []
            disagreement_points = "none"
            delta_enabled_now = self.now_round >= 2
            delta_leader_report, has_compact_leader_report = self._leader_report_for_delta()
            if delta_enabled_now:
                disagreement_points = self._extract_disagreement_points(prev_round_responses)

            for doctor_agent in self.doctor_agents:
                doctor_agent.current_round = self.now_round
                doctor_agent.delta_disagreement_points = disagreement_points
                doctor_agent.delta_leader_report_compact = delta_leader_report

            def process_doctoragents(doctor_agent, i):
                response, messages, prompt_token, completion_token = doctor_agent.collaboration(self.leader_opinion, self.leader_report)
                self.doctor_responses.append(response)
                json.dump(response, open(f"{self.save_dir}/doctor{i + 1}_round{self.now_round}_review.json", "w"))
                json.dump(messages, open(f"{self.save_dir}/doctor{i + 1}_round{self.now_round}_review_messages.json", "w"))
                with open(f"{self.save_dir}/doctor{i + 1}_round{self.now_round}_review_userprompt.txt", "w") as f:
                    f.write(messages[1]["content"])
                with open(f"{self.save_dir}/doctor{i + 1}_round{self.now_round}_delta_discussion_log.json", "w") as f:
                    json.dump(
                        {
                            "round_number": int(response.get("round_number", self.now_round)),
                            "delta_discussion_enabled": delta_enabled_now,
                            "disagreement_points": str(response.get("disagreement_points", "none")),
                            "used_compact_doctor_text": bool(response.get("used_compact_doctor_text", False)),
                            "used_compact_leader_report": bool(response.get("used_compact_leader_report", False)) and has_compact_leader_report,
                            "delta_fallback_used": bool(response.get("delta_fallback_used", False)) or (delta_enabled_now and self._is_none_text(str(response.get("disagreement_points", "none")))),
                            "retrieval_context_length_chars": int(response.get("retrieval_context_length_chars", 0)),
                            "non_retrieval_payload_length_chars": int(response.get("non_retrieval_payload_length_chars", 0)),
                            "collaboration_user_prompt_length_chars": int(response.get("collaboration_user_prompt_length_chars", 0)),
                            "collaboration_prompt_token": int(response.get("collaboration_prompt_token", 0)),
                            "collaboration_completion_token": int(response.get("collaboration_completion_token", 0)),
                            "compact_doctor_text_length_chars": int(response.get("compact_doctor_text_length_chars", 0)),
                            "compact_leader_report_length_chars": int(response.get("compact_leader_report_length_chars", 0)),
                            "disagreement_points_length_chars": int(response.get("disagreement_points_length_chars", 0)),
                            "collaboration_prompt_preview": (messages[1]["content"] if len(messages) > 1 else "")[:800],
                        },
                        f,
                    )
                return response, prompt_token, completion_token

            with ThreadPoolExecutor(max_workers=self.doctor_num) as executor:
                futures = [executor.submit(process_doctoragents, doctor_agent, i) for i, doctor_agent in enumerate(self.doctor_agents)]

                for future in futures:
                    response, prompt_token, completion_token = future.result()
                    prompt_tokens += prompt_token
                    completion_tokens += completion_token
                    efficiency_records.append(
                        {
                            "round_number": int(response.get("round_number", self.now_round)),
                            "collaboration_prompt_token": int(response.get("collaboration_prompt_token", 0)),
                            "collaboration_completion_token": int(response.get("collaboration_completion_token", 0)),
                            "collaboration_user_prompt_length_chars": int(response.get("collaboration_user_prompt_length_chars", 0)),
                            "non_retrieval_payload_length_chars": int(response.get("non_retrieval_payload_length_chars", 0)),
                        }
                    )

                # for i, doctor in enumerate(self.doctor_agents):
                #     response, messages, prompt_token, completion_token = doctor.collaboration(
                #         self.leader_opinion, self.leader_report)
                #     self.doctor_responses.append(response)
                #     json.dump(response, open(
                #         f"{self.save_dir}/doctor{i + 1}_round{self.now_round}_review.json", "w"))
                #     json.dump(messages, open(
                #         f"{self.save_dir}/doctor{i + 1}_round{self.now_round}_review_messages.json", "w"))
                #     with open(f"{self.save_dir}/doctor{i + 1}_round{self.now_round}_review_userprompt.txt", "w") as f:
                #         f.write(messages[1]["content"])
                #     prompt_tokens += prompt_token
                #     completion_tokens += completion_token

            action, summary_content, messages, prompt_token, completion_token = self.leader_agent.summary(self.doctor_responses)
            json.dump(summary_content, open(f"{self.save_dir}/leader_round{self.now_round}_summary.json", "w"))
            json.dump(messages, open(f"{self.save_dir}/leader_round{self.now_round}_summary_messages.json", "w"))
            with open(f"{self.save_dir}/leader_round{self.now_round}_compact_report_log.json", "w") as f:
                json.dump(
                    {
                        "leader_report_compact_text_preview": str(summary_content.get("leader_report_compact_text", summary_content.get("report", "")))[:800],
                        "compact_leader_enabled": True,
                        "report_source": summary_content.get("report_source", "fallback"),
                    },
                    f,
                )
            prompt_tokens += prompt_token
            completion_tokens += completion_token
            self.leader_opinion = summary_content["answer"]
            self.leader_report = summary_content["report"]
            self.last_leader_report_fields = self._parse_compact_leader_report_text(
                summary_content.get("leader_report_compact_text", self.leader_report)
            )

            if action == 0:
                break
            print("continue to next round")

        if self.now_round == self.max_round:
            print("No agreement has been reached")
        num_rounds_executed = int(self.now_round)

        def _avg_for_round(records: List[Dict], round_number: int, key: str):
            vals = [float(r.get(key, 0.0)) for r in records if int(r.get("round_number", -1)) == round_number]
            if not vals:
                return None
            return sum(vals) / len(vals)

        sample_id = os.path.basename(self.save_dir)
        summary_payload = {
            "sample_id": str(sample_id),
            "num_rounds_executed": num_rounds_executed,
            "round1_avg_prompt_token": _avg_for_round(efficiency_records, 1, "collaboration_prompt_token"),
            "round2_avg_prompt_token": _avg_for_round(efficiency_records, 2, "collaboration_prompt_token"),
            "round1_avg_user_prompt_length_chars": _avg_for_round(efficiency_records, 1, "collaboration_user_prompt_length_chars"),
            "round2_avg_user_prompt_length_chars": _avg_for_round(efficiency_records, 2, "collaboration_user_prompt_length_chars"),
            "round1_avg_non_retrieval_payload_length_chars": _avg_for_round(efficiency_records, 1, "non_retrieval_payload_length_chars"),
            "round2_avg_non_retrieval_payload_length_chars": _avg_for_round(efficiency_records, 2, "non_retrieval_payload_length_chars"),
        }
        with open(f"{self.save_dir}/discussion_efficiency_summary.json", "w") as f:
            json.dump(summary_payload, f)

        self.now_round = 0
        return self.leader_report, self.now_round, prompt_tokens, completion_tokens


def extract_and_parse_json(text: str):
    def _try_parse(candidate: str):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                obj, _ = json.JSONDecoder().raw_decode(candidate)
                return obj
            except Exception:
                return None

    pattern_backticks = r'```json(.*?)```'
    match = re.search(pattern_backticks, text, re.DOTALL)
    if match:
        json_string = match.group(1).strip().replace("\n", "")
        parsed = _try_parse(json_string)
        if parsed is not None:
            return parsed

    match = re.search(pattern_backticks, text.strip() + "\"]}```", re.DOTALL)
    if match:
        json_string = match.group(1).strip().replace("\n", "")
        parsed = _try_parse(json_string)
        if parsed is not None:
            return parsed

    pattern_json_object = r'\{.*?\}'
    match = re.search(pattern_json_object, text, re.DOTALL)
    if match:
        json_string = match.group(0).strip().replace("\n", "")
        parsed = _try_parse(json_string)
        if parsed is not None:
            return parsed

    # Fallback: return raw text to avoid hard failure when LLM response is not valid JSON.
    return {"raw": text.strip()}
