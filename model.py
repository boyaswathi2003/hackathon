from typing import Dict, List, Optional
from utils import load_db, age_group, parse_prescription
from dataclasses import dataclass
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from config import GRANITE_MODEL_ID, HF_TOKEN

@dataclass
class DrugRecord:
    name: str
    adult_dose: str
    child_dose: str
    aliases: List[str]

class DrugDB:
    def __init__(self, path: str = "datasets/drug_data.json"):
        self.db = load_db(path)

    def list_drugs(self) -> List[str]:
        return [d["name"] for d in self.db["drugs"]]

    def normalize(self, name: str) -> Optional[str]:
        n = name.lower().strip()
        for d in self.db["drugs"]:
            if d["name"].lower() == n or n in [a.lower() for a in d.get("aliases", [])]:
                return d["name"]
        return None

    def default_dose_for_age(self, drug: str, age: int) -> Optional[str]:
        d = next((x for x in self.db["drugs"] if x["name"] == drug), None)
        if not d: return None
        grp = age_group(age)
        if grp in ("child", "adolescent"):
            return d.get("child_dose")
        return d.get("adult_dose")

    def max_daily_mg(self, drug: str, age: int) -> Optional[int]:
        m = self.db.get("max_daily_dose_mg", {}).get(drug)
        if not m: return None
        grp = age_group(age)
        if grp in ("child", "adolescent"):
            return m.get("child")
        return m.get("adult")

    def interactions_for(self, drugs: List[str]) -> List[Dict]:
        pairs = set()
        out = []
        for a in drugs:
            for b in drugs:
                if a >= b:  # avoid duplicates
                    continue
                key = (a,b)
                if key in pairs: continue
                pairs.add(key)
                for it in self.db["interactions"]:
                    p = it["pair"]
                    if set(p) == set([a,b]):
                        out.append(it)
        return out

    def alternatives(self, drug: str) -> List[str]:
        return self.db.get("alternatives", {}).get(drug, [])

class GraniteClient:
    """
    IBM Granite models from Hugging Face.
    Used for NLP tasks like extracting structured drug info.
    """
    def __init__(self, model_id: str = GRANITE_MODEL_ID, hf_token: str = HF_TOKEN):
        self.model_id = model_id
        self.hf_token = hf_token
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                use_auth_token=hf_token,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        except Exception as e:
            print(f"Granite model load failed: {e}")
            self.pipe = None

    def extract_drug_info(self, text: str):
        """
        Ask Granite model to extract structured drug info (drug, dose, frequency).
        """
        if not self.pipe:
            return []

        prompt = f"""
        Extract medicines from the following prescription text.
        Return JSON list with keys: drug, dose_mg, frequency_per_day.

        Prescription: {text}
        """
        try:
            response = self.pipe(prompt, max_new_tokens=200, do_sample=False)[0]['generated_text']
            return response
        except Exception as e:
            return {"error": str(e)}

class Analyzer:
    def __init__(self, db_path: str = "datasets/drug_data.json"):
        self.db = DrugDB(db_path)
        self.granite = GraniteClient()

    def extract(self, text: Optional[str], explicit_drugs: Optional[List[Dict]]) -> List[Dict]:
        if explicit_drugs and len(explicit_drugs) > 0:
            cleaned = []
            for d in explicit_drugs:
                nm = self.db.normalize(d.get("drug","")) or d.get("drug","")
                cleaned.append({
                    "drug": nm,
                    "dose_mg": d.get("dose_mg"),
                    "frequency_per_day": d.get("frequency_per_day")
                })
            return cleaned
        if text:
            granite_result = self.granite.extract_drug_info(text)
            if granite_result and isinstance(granite_result, list):
                return granite_result
            # fallback regex parser
            return parse_prescription(text, self.db.db)
        return []

    def check(self, items: List[Dict], age: int) -> Dict:
        drugs = [i["drug"] for i in items]
        interactions = self.db.interactions_for(drugs)

        recs = {}
        warnings = []
        for it in items:
            nm = it["drug"]
            dose = it.get("dose_mg")
            freq = it.get("frequency_per_day")
            recommended = self.db.default_dose_for_age(nm, age)
            max_daily = self.db.max_daily_mg(nm, age)
            total = dose * freq if dose and freq else None
            if total and max_daily and total > max_daily:
                warnings.append({
                    "drug": nm,
                    "issue": "Dose exceeds max daily limit",
                    "computed_mg_per_day": total,
                    "max_daily_mg": max_daily
                })
            recs[nm] = {
                "recommended_dose_for_age": recommended,
                "max_daily_mg": max_daily
            }

        alts = {nm: self.db.alternatives(nm) for nm in drugs}

        return {
            "drugs_parsed": items,
            "interactions": interactions,
            "dosage_guidance": recs,
            "warnings": warnings,
            "alternatives": alts
        }
