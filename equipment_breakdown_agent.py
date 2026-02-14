"""
equipment_breakdown_agent.py

Lightweight predictive-maintenance agent that uses AWS Bedrock (via boto3)
to analyze equipment data (SAP PM / SCM / PP exports) and return
probabilities of breakdown plus recommended actions.

This script is intentionally self-contained and falls back to a simple
rule-based predictor if Bedrock credentials/connection are not available.

Usage:
  - Prepare a CSV with equipment events/attributes. Expected columns (best-effort):
      equipment_id, last_maintenance_days, failure_count, avg_daily_usage, temperature, spare_parts_on_hand
  - Run: python equipment_breakdown_agent.py --csv path/to/data.csv

Outputs a CSV-like text of predictions and prints recommended actions.
"""
import argparse
import os
import csv
import io
import json
from typing import Optional

try:
    import pandas as pd
except Exception:
    pd = None

import boto3
from botocore.config import Config
import pickle
import numpy as np
import glob

# ---------------- CONFIG ----------------
REGION = os.environ.get("AWS_REGION", "us-east-1")
# Pick a reasonable model; adjust to your account/region. This example uses Claude Haiku as in the repo.
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

BEDROCK_VERIFY = None  # path to corporate cert if needed, e.g. r"C:\path\to\cert.crt"

INFERENCE_CONFIG = {"maxTokens": 1024, "temperature": 0.2}


def discover_local_artifacts(models_dir: str = None):
    """Look for a saved local model and scaler in ./models (or provided dir).
    Returns (model_path, scaler_path) or (None, None).
    """
    if models_dir is None:
        base = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base, 'models')
    if not os.path.isdir(models_dir):
        return None, None
    candidates = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.startswith('maintenance_model_') and f.endswith('.pkl')]
    if not candidates:
        return None, None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    model_path = candidates[0]
    scaler_path = os.path.join(models_dir, 'data_scaler.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = None
    return model_path, scaler_path


class EquipmentBreakdownAgent:
    def __init__(self, region: str = REGION, model_id: str = MODEL_ID, verify: Optional[str] = BEDROCK_VERIFY,
                 local_model_path: Optional[str] = None, local_scaler_path: Optional[str] = None, use_bedrock: bool = False):
        self.region = region
        self.model_id = model_id
        self.verify = verify
        # Lazy init Bedrock client
        self._client = None
        # Local model artifacts
        self.local_model_path = local_model_path
        self.local_scaler_path = local_scaler_path
        self.local_model = None
        self.local_scaler = None
        self.use_bedrock = use_bedrock

    @property
    def client(self):
        if self._client is None:
            try:
                self._client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=self.region,
                    verify=self.verify
                )
            except Exception as e:
                print(f"Warning: could not create Bedrock client: {e}")
                self._client = None
        return self._client

    def load_csv(self, path: Optional[str]):
        if path is None:
            return None
        if pd:
            return pd.read_csv(path)
        # Fallback: parse with csv module
        data = []
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

    def load_local_model(self):
        """Load local pickled model and scaler if paths were provided."""
        if self.local_model is None and self.local_model_path:
            try:
                with open(self.local_model_path, 'rb') as f:
                    self.local_model = pickle.load(f)
                print(f"Loaded local model from {self.local_model_path}")
            except Exception as e:
                print(f"Failed to load local model: {e}")
                self.local_model = None

        if self.local_scaler is None and self.local_scaler_path:
            try:
                with open(self.local_scaler_path, 'rb') as f:
                    self.local_scaler = pickle.load(f)
                print(f"Loaded local scaler from {self.local_scaler_path}")
            except Exception as e:
                print(f"Failed to load local scaler: {e}")
                self.local_scaler = None

    def _align_features_for_model(self, df):
        """Attempt to prepare a numeric feature matrix compatible with the local model.
        This mirrors the basic logic of Model_Python.py but is intentionally forgiving.
        Returns (X_df, feature_names) or (None, None) on failure.
        """
        if pd and isinstance(df, pd.DataFrame):
            X = df.copy()
        elif isinstance(df, list):
            X = pd.DataFrame(df)
        else:
            return None, None

        # drop identifier/time columns if present
        for dropc in ['equipment_id', 'Equipment_ID', 'Timestamp', 'timestamp']:
            if dropc in X.columns:
                X = X.drop(columns=[dropc])

        # select numeric columns only (the model was trained on numeric features)
        X_num = X.select_dtypes(include=[np.number])
        if X_num.shape[1] == 0:
            # try to coerce object columns to numeric where possible
            for c in X.columns:
                X[c] = pd.to_numeric(X[c], errors='coerce')
            X_num = X.select_dtypes(include=[np.number])

        if X_num.shape[1] == 0:
            return None, None

        return X_num, list(X_num.columns)

    def predict_with_local_model(self, df):
        """Run the local model (if loaded) and return a CSV string with predictions and actions."""
        self.load_local_model()
        if self.local_model is None:
            return None

        X_df, feature_names = self._align_features_for_model(df)
        if X_df is None:
            print("Could not prepare numeric features for the local model.")
            return None

        X = X_df.values
        # scale if scaler available
        if self.local_scaler is not None:
            try:
                X = self.local_scaler.transform(X)
            except Exception as e:
                print(f"Scaler transform failed: {e}")

        # get probabilities if possible
        prob_col = None
        try:
            if hasattr(self.local_model, 'predict_proba'):
                probs = self.local_model.predict_proba(X)
                # if binary, take class 1 probability
                if probs.shape[1] == 2:
                    prob_col = probs[:, 1]
                else:
                    # fallback: take max probability across classes
                    prob_col = probs.max(axis=1)
            else:
                preds = self.local_model.predict(X)
                # fallback map: 1 -> 1.0, 0 -> 0.0
                prob_col = [float(p) for p in preds]
        except Exception as e:
            print(f"Local model prediction failed: {e}")
            return None

        # build rows with recommended actions
        out_rows = []
        ids = []
        # try to get equipment ids from original df
        if pd and isinstance(df, pd.DataFrame):
            if 'equipment_id' in df.columns:
                ids = list(df['equipment_id'].astype(str))
            elif 'Equipment_ID' in df.columns:
                ids = list(df['Equipment_ID'].astype(str))
        if not ids:
            ids = [f"row{i+1}" for i in range(len(prob_col))]

        for eq, p in zip(ids, prob_col):
            prob = float(p)
            actions = self._recommend_actions_from_prob(prob)
            # simple top failure modes heuristic
            if prob > 0.6:
                modes = 'bearing_wear;hydraulic_leak'
            elif prob > 0.25:
                modes = 'bearing_wear'
            else:
                modes = 'none'
            out_rows.append({
                'equipment_id': eq,
                'probability_breakdown': f"{prob:.2f}",
                'top_failure_modes': modes,
                'recommended_actions': ';'.join(actions)
            })

        # convert to CSV
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=['equipment_id','probability_breakdown','top_failure_modes','recommended_actions'])
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
        return out.getvalue()

    def parse_prediction_csv(self, csv_text: str):
        """Parse a prediction CSV produced by this agent into a list of dicts.
        Each dict contains equipment_id, probability_breakdown (float), top_failure_modes, recommended_actions.
        """
        if not csv_text:
            return []
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = []
        for r in reader:
            try:
                prob = float(r.get('probability_breakdown', 0.0))
            except Exception:
                # try to strip and parse
                try:
                    prob = float(str(r.get('probability_breakdown','0')).strip())
                except Exception:
                    prob = 0.0
            rows.append({
                'equipment_id': r.get('equipment_id'),
                'probability_breakdown': prob,
                'top_failure_modes': r.get('top_failure_modes',''),
                'recommended_actions': r.get('recommended_actions','')
            })
        return rows

    def explain_with_bedrock(self, question: str, data, predictions_csv: str):
        """Ask Bedrock to answer a question using the data sample and predictions as context."""
        sample = self._prepare_sample(data)
        # Build prompt that includes the predictions CSV as context
        system = (
            "You are an industrial maintenance assistant. Use the provided data and prediction CSV to answer the user's question concisely. "
            "If the question requests actionable steps, prioritize and be specific. If information is missing, say so."
        )
        human = (
            f"Data sample:\n{sample}\n\nPredictions:\n{predictions_csv}\n\nQuestion: {question}\n\n"
            "Answer in plain text; do not return CSV unless explicitly asked."
        )
        messages = [
            {"role": "system", "content": [{"text": system}]},
            {"role": "user", "content": [{"text": human}]}
        ]
        try:
            resp = self.call_bedrock(messages)
            return resp
        except Exception as e:
            print(f"Bedrock explanation failed: {e}")
            return None

    def answer_question(self, question: str, data, predictions_csv: str):
        """Answer a user's question. Prefer Bedrock if enabled; otherwise use heuristics over predictions."""
        q = (question or '').strip()
        if not q:
            return "No question provided."

        # Parse predictions for heuristics
        preds = self.parse_prediction_csv(predictions_csv)

        # If Bedrock is enabled, try it first for rich answers
        if self.use_bedrock and self.client is not None:
            ans = self.explain_with_bedrock(q, data, predictions_csv)
            if ans:
                return ans

        # Heuristic fallbacks
        ql = q.lower()
        # Which equipment at highest risk
        if ('which' in ql and ('risk' in ql or 'at risk' in ql or 'high risk' in ql)) or 'top' in ql and 'risk' in ql:
            if not preds:
                return "No prediction results available to determine top-risk equipment."
            sorted_preds = sorted(preds, key=lambda r: r['probability_breakdown'], reverse=True)
            top = sorted_preds[:5]
            lines = [f"{p['equipment_id']}: {p['probability_breakdown']:.2f}" for p in top]
            return "Top at-risk equipment:\n" + "\n".join(lines)

        # Actions recommendation aggregation
        if 'what' in ql and ('action' in ql or 'recommend' in ql or 'recommendation' in ql):
            if not preds:
                return "No prediction results available to produce recommendations."
            # aggregate actions for high risk items
            high = [p for p in preds if p['probability_breakdown'] >= 0.5]
            actions = set()
            for h in high:
                for a in str(h.get('recommended_actions','')).split(';'):
                    if a:
                        actions.add(a)
            if not actions:
                return "No specific actions required based on current predictions."
            return "Recommended actions for high-risk items:\n" + "\n".join(actions)

        # Ask for a specific equipment id
        # e.g., "probability for EQ100" or "what about EQ200"
        for p in preds:
            eid = str(p.get('equipment_id','')).lower()
            if eid and eid in ql:
                return (f"Equipment {p['equipment_id']} - probability of breakdown: {p['probability_breakdown']:.2f}. "
                        f"Top failure modes: {p.get('top_failure_modes','')}. Actions: {p.get('recommended_actions','')}")

        # Default fallback: summarise average risk
        if preds:
            avg = sum(p['probability_breakdown'] for p in preds) / max(1, len(preds))
            return f"Average predicted probability across provided items: {avg:.2f}. Ask 'which are at risk' or 'what actions' for more detail."

        return "I couldn't answer that question with the available data. Try a different phrasing or enable Bedrock with --use-bedrock."

    def _recommend_actions_from_prob(self, prob: float):
        if prob > 0.6:
            return ['schedule_immediate_maintenance', 'inspect_top_failure_modes', 'order_critical_spares']
        elif prob > 0.25:
            return ['inspect_during_next_shift', 'increase_monitoring', 'check_spare_inventory']
        else:
            return ['continue_routine_maintenance']

    def _prepare_sample(self, data, max_rows: int = 10):
        """Return a compact CSV string of up to max_rows rows for sending to the LLM."""
        if data is None:
            return ""
        if pd and isinstance(data, pd.DataFrame):
            sample = data.head(max_rows)
            return sample.to_csv(index=False)
        # list/dict fallback
        if isinstance(data, list):
            if not data:
                return ""
            fieldnames = list(data[0].keys())
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for row in data[:max_rows]:
                writer.writerow(row)
            return output.getvalue()
        return str(data)

    def _build_prompt(self, sample_csv: str, request: str):
        system = (
            "You are an industrial predictive-maintenance assistant. "
            "Given tabular equipment data, predict the probability of equipment breakdown within the next 30 days. "
            "For each equipment row return a CSV with header: equipment_id,probability_breakdown,top_failure_modes,recommended_actions. "
            "Probabilities must be numbers between 0 and 1 with two decimal places. "
            "Recommended actions should be short, actionable, and prioritized. "
        )

        # Provide a tiny example to steer output formatting
        example_user = (
            "Example input CSV:\n" +
            "equipment_id,last_maintenance_days,failure_count,avg_daily_usage,temperature,spare_parts_on_hand\n"
            "EQ100,120,2,24,85,0\n"
            "EQ200,10,0,8,60,5\n"
            "\nExpected output CSV:\n"
            "equipment_id,probability_breakdown,top_failure_modes,recommended_actions\n"
            "EQ100,0.85,hydraulic_leak;bearing_wear,inspect_hydraulics;order_bearing;schedule_urgent_maintenance\n"
            "EQ200,0.05,none,continue_routine_maintenance\n"
        )

        human = f"Data sample:\n{sample_csv}\n\nRequest: {request}\n\nReturn only the CSV result, no extra commentary."

        messages = [
            {"role": "system", "content": [{"text": system}]},
            {"role": "user", "content": [{"text": example_user + "\n" + human}]}
        ]
        return messages

    def call_bedrock(self, messages):
        client = self.client
        if client is None:
            raise RuntimeError("Bedrock client not available")
        try:
            response = client.converse(
                modelId=self.model_id,
                messages=messages,
                inferenceConfig=INFERENCE_CONFIG
            )
            # Extract text (same pattern used in the repo)
            return response['output']['message']['content'][0]['text']
        except Exception as e:
            raise

    def parse_csv_text(self, text: str):
        try:
            # Find start of CSV (in case LLM adds markdown)
            start = 0
            # remove markdown fences if present
            text = text.strip()
            if text.startswith('```'):
                lines = text.splitlines()
                # skip first fence
                lines = [ln for ln in lines if not ln.startswith('```')]
                text = '\n'.join(lines).strip()
            return text
        except Exception:
            return text

    def rule_based_predict(self, df):
        """A simple heuristic fallback predictor: returns CSV string."""
        rows = []
        def safe_float(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return default

        if pd and isinstance(df, pd.DataFrame):
            iter_rows = df.to_dict(orient='records')
        elif isinstance(df, list):
            iter_rows = df
        else:
            return ""

        for r in iter_rows:
            eq = r.get('equipment_id') or r.get('id') or 'unknown'
            last_maint = safe_float(r.get('last_maintenance_days', 0))
            failures = safe_float(r.get('failure_count', 0))
            usage = safe_float(r.get('avg_daily_usage', 0))
            temp = safe_float(r.get('temperature', 0))
            parts = safe_float(r.get('spare_parts_on_hand', 0))

            score = 0.0
            # heuristics
            score += min(last_maint / 365.0, 1.0) * 0.4
            score += min(failures / 5.0, 1.0) * 0.3
            score += min(usage / 24.0, 1.0) * 0.2
            score += max(0.0, (temp - 70) / 50.0) * 0.1
            # reduce risk if spare parts available
            if parts > 3:
                score *= 0.7
            prob = max(0.0, min(1.0, round(score, 2)))
            modes = []
            if failures >= 2:
                modes.append('bearing_wear')
            if temp > 85:
                modes.append('overheating')
            if parts < 1:
                modes.append('spare_part_shortage')
            if not modes:
                modes = ['unknown']
            actions = []
            if prob > 0.6:
                actions = ['schedule_immediate_maintenance', 'inspect_top_failure_modes', 'order_critical_spares']
            elif prob > 0.25:
                actions = ['inspect_during_next_shift', 'increase_monitoring', 'check_spare_inventory']
            else:
                actions = ['continue_routine_maintenance']

            rows.append({
                'equipment_id': eq,
                'probability_breakdown': f"{prob:.2f}",
                'top_failure_modes': ';'.join(modes),
                'recommended_actions': ';'.join(actions)
            })

        # convert to CSV
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=['equipment_id','probability_breakdown','top_failure_modes','recommended_actions'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return out.getvalue()

    def predict(self, data, request: str = "Predict breakdown probability for next 30 days"):
        # Prefer a local model prediction if available
        local_csv = None
        try:
            local_csv = self.predict_with_local_model(data)
        except Exception as e:
            print(f"Local model prediction raised an exception: {e}")

        if local_csv:
            return local_csv

        # If configured to use Bedrock, try the LLM path
        if self.use_bedrock:
            sample = self._prepare_sample(data)
            messages = self._build_prompt(sample, request)
            try:
                txt = self.call_bedrock(messages)
                csv_text = self.parse_csv_text(txt)
                return csv_text
            except Exception as e:
                print(f"Bedrock call failed or unavailable: {e}\nFalling back to rule-based predictor.")

        # final fallback: rule-based
        return self.rule_based_predict(data)


def main():
    parser = argparse.ArgumentParser(description='Equipment breakdown prediction agent (Bedrock-backed with fallback).')
    parser.add_argument('--csv', help='Path to equipment CSV export (optional).')
    parser.add_argument('--model-id', help='Bedrock model id (overrides env/BEDROCK_MODEL_ID).')
    parser.add_argument('--local-model', help='Path to local pickled model (overrides LLM).')
    parser.add_argument('--scaler', help='Path to local pickled scaler (optional, used to transform features).')
    parser.add_argument('--use-bedrock', action='store_true', help='Enable Bedrock LLM fallback/explanation (if no local model).')
    parser.add_argument('--interactive', action='store_true', help='Start interactive Q&A session after predictions are produced.')
    parser.add_argument('--output', help='Path to save prediction CSV output (default: predictions_output.csv)', default='predictions_output.csv')
    parser.add_argument('--equipment-id', help='Equipment ID or comma-separated list of IDs to predict for (default: all).')
    parser.add_argument('--equipment-json', help='JSON string or path to a JSON/CSV file containing a single equipment record to predict for.')
    args = parser.parse_args()

    # If no input flags were provided, prompt the user to supply equipment details or a CSV path
    if not (args.csv or args.equipment_json or args.equipment_id or args.interactive):
        try:
            want = input("No input provided. Do you want to provide equipment details now? (y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            want = 'n'
        if want == 'y':
            try:
                opt = input("Type 'json' to paste equipment JSON, 'csv' to provide CSV path, 'id' to enter equipment id(s): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                opt = ''
            if opt == 'json':
                try:
                    text = input("Paste equipment JSON object or a JSON file path: ").strip()
                except (EOFError, KeyboardInterrupt):
                    text = ''
                if text:
                    args.equipment_json = text
            elif opt == 'csv':
                try:
                    path = input("Enter CSV file path: ").strip()
                except (EOFError, KeyboardInterrupt):
                    path = ''
                if path:
                    args.csv = path
            elif opt == 'id':
                try:
                    ids = input("Enter equipment id(s), comma-separated: ").strip()
                except (EOFError, KeyboardInterrupt):
                    ids = ''
                if ids:
                    args.equipment_id = ids

    # If user did not supply local model paths, attempt to auto-discover artifacts in ./models/
    if not args.local_model:
        found_model, found_scaler = discover_local_artifacts()
        if found_model:
            print(f"Discovered local model: {found_model}")
            args.local_model = found_model
            if found_scaler:
                print(f"Discovered scaler: {found_scaler}")
                args.scaler = found_scaler

    model_id = args.model_id or os.environ.get('BEDROCK_MODEL_ID') or MODEL_ID
    agent = EquipmentBreakdownAgent(region=REGION, model_id=model_id, verify=BEDROCK_VERIFY,
                                    local_model_path=args.local_model, local_scaler_path=args.scaler,
                                    use_bedrock=args.use_bedrock)

    data = None
    if args.csv:
        data = agent.load_csv(args.csv)
    # If user supplied a single equipment via JSON (string or file path), prefer that
    if args.equipment_json:
        raw = args.equipment_json
        # If it's a path to a file
        if os.path.exists(raw):
            # support .json or .csv
            if raw.lower().endswith('.json'):
                with open(raw, 'r', encoding='utf-8') as f:
                    try:
                        obj = json.load(f)
                        if isinstance(obj, list):
                            data = obj
                        elif isinstance(obj, dict):
                            data = [obj]
                    except Exception as e:
                        print(f"Failed to parse JSON file {raw}: {e}")
            elif raw.lower().endswith('.csv'):
                # load csv and take first row
                if pd:
                    df_single = pd.read_csv(raw)
                    data = df_single.head(1)
                else:
                    with open(raw, newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        rows = [r for r in reader]
                        if rows:
                            data = [rows[0]]
        else:
            # try to parse as JSON string
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    data = [obj]
                elif isinstance(obj, list):
                    data = obj
            except Exception:
                # not JSON; leave as-is and will be handled by equipment-id flow
                pass
    else:
        # small synthetic example
        sample_csv = (
            "equipment_id,last_maintenance_days,failure_count,avg_daily_usage,temperature,spare_parts_on_hand\n"
            "EQ100,200,3,20,90,0\n"
            "EQ200,15,0,6,60,10\n"
            "EQ300,45,1,12,75,2\n"
        )
        # If pandas available, load into DataFrame
        if pd:
            data = pd.read_csv(io.StringIO(sample_csv))
        else:
            reader = csv.DictReader(io.StringIO(sample_csv))
            data = [r for r in reader]

    print("\n--- Running prediction pipeline ---\n")
    # If user supplied equipment ids via CLI, or asked to select interactively, filter data
    def filter_data_by_equipment(data_obj, eq_ids):
        if not eq_ids:
            return data_obj
        # normalize ids set
        ids_set = set([e.strip().lower() for e in eq_ids if e and e.strip()])
        if pd and isinstance(data_obj, pd.DataFrame):
            col = None
            if 'equipment_id' in data_obj.columns:
                col = 'equipment_id'
            elif 'Equipment_ID' in data_obj.columns:
                col = 'Equipment_ID'
            if col is None:
                return data_obj
            mask = data_obj[col].astype(str).str.lower().isin(ids_set)
            return data_obj[mask]
        elif isinstance(data_obj, list):
            out = [r for r in data_obj if str(r.get('equipment_id') or r.get('Equipment_ID') or '').strip().lower() in ids_set]
            return out
        return data_obj

    # Determine equipment ids from args or interactive prompt
    equipment_ids = None
    if args.equipment_id:
        # allow comma-separated list
        equipment_ids = [s.strip() for s in args.equipment_id.split(',') if s.strip()]
    elif args.interactive:
        try:
            sel = input("Enter equipment id(s) to predict (comma-separated) or 'all' for all items: ").strip()
        except (EOFError, KeyboardInterrupt):
            sel = ''
        if sel and sel.lower() != 'all':
            # If user pasted JSON object or provided a file path, try to parse as a single equipment
            if sel.startswith('{') or sel.startswith('[') or os.path.exists(sel):
                # attempt to parse JSON or file
                parsed = None
                if os.path.exists(sel):
                    path = sel
                    if path.lower().endswith('.json'):
                        try:
                            with open(path, 'r', encoding='utf-8') as f:
                                parsed = json.load(f)
                        except Exception as e:
                            print(f"Failed to read JSON file {path}: {e}")
                    elif path.lower().endswith('.csv'):
                        if pd:
                            try:
                                parsed_df = pd.read_csv(path)
                                parsed = parsed_df.head(1).to_dict(orient='records')
                            except Exception as e:
                                print(f"Failed to read CSV file {path}: {e}")
                        else:
                            try:
                                with open(path, newline='', encoding='utf-8') as f:
                                    reader = csv.DictReader(f)
                                    rows = [r for r in reader]
                                    if rows:
                                        parsed = [rows[0]]
                            except Exception as e:
                                print(f"Failed to read CSV file {path}: {e}")
                else:
                    try:
                        obj = json.loads(sel)
                        parsed = obj if obj is not None else None
                    except Exception:
                        parsed = None

                if parsed:
                    # set data to parsed (dict or list)
                    if isinstance(parsed, dict):
                        data = [parsed]
                    else:
                        data = parsed
                else:
                    equipment_ids = [s.strip() for s in sel.split(',') if s.strip()]
            else:
                equipment_ids = [s.strip() for s in sel.split(',') if s.strip()]

    if equipment_ids:
        filtered = filter_data_by_equipment(data, equipment_ids)
        # if nothing matched, warn and exit
        if (pd and isinstance(filtered, pd.DataFrame) and filtered.shape[0] == 0) or (isinstance(filtered, list) and len(filtered) == 0):
            print(f"No matching equipment rows found for IDs: {equipment_ids}. Exiting.")
            return
        data = filtered

    result_csv = agent.predict(data)
    print("--- Predictions (CSV) ---")
    print(result_csv)

    # Optionally save results
    out_path = args.output
    try:
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            f.write(result_csv)
        print(f"Predictions saved to {out_path}")
    except Exception as e:
        print(f"Failed to save predictions to {out_path}: {e}")

    # Interactive prompt: allow the user to provide equipment details for direct prediction and to ask follow-up questions
    if args.interactive:
        # Optionally accept a new equipment record directly from the user and predict for it
        try:
            want = input("Do you want to provide equipment details to predict now? (y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            want = 'n'
        if want == 'y':
            try:
                choice = input("Paste equipment JSON/object now, or type 'csv' to select from loaded CSV: ").strip()
            except (EOFError, KeyboardInterrupt):
                choice = ''
            provided_data = None
            if choice.lower() == 'csv':
                try:
                    eqs = input("Enter equipment id(s) (comma-separated) to predict: ").strip()
                except (EOFError, KeyboardInterrupt):
                    eqs = ''
                if eqs:
                    ids = [s.strip() for s in eqs.split(',') if s.strip()]
                    try:
                        filtered = filter_data_by_equipment(data, ids)
                    except Exception:
                        filtered = data
                    provided_data = filtered
            else:
                text = choice
                if not text:
                    try:
                        text = input("Paste equipment JSON object now: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        text = ''
                if text:
                    try:
                        obj = json.loads(text)
                        if isinstance(obj, dict):
                            provided_data = [obj]
                        elif isinstance(obj, list):
                            provided_data = obj
                    except Exception as e:
                        print(f"Failed to parse JSON input: {e}")
                        provided_data = None

            if provided_data:
                # If agent has no local_model_path, attempt to discover saved artifacts
                if not agent.local_model_path:
                    found_model, found_scaler = discover_local_artifacts()
                    if found_model:
                        agent.local_model_path = found_model
                        agent.local_scaler_path = found_scaler

                # Predict using agent (it will prefer local model if available)
                try:
                    csv_single = agent.predict(provided_data)
                except Exception as e:
                    print(f"Prediction failed: {e}")
                    csv_single = None

                if csv_single:
                    parsed = agent.parse_prediction_csv(csv_single)
                    if parsed and parsed[0].get('equipment_id'):
                        name = parsed[0]['equipment_id']
                    else:
                        name = 'equipment'
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    fname = f"{name}_{ts}.csv"
                    try:
                        with open(fname, 'w', encoding='utf-8', newline='') as f:
                            f.write(csv_single)
                        print(f"Saved prediction for provided equipment to {fname}")
                    except Exception as e:
                        print(f"Failed to write prediction file {fname}: {e}")

        # After optional direct prediction, continue into Q&A loop
        print("\n--- Interactive Q&A (type 'exit' or 'quit' to stop) ---")
        while True:
            try:
                question = input("Question about the predictions> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting interactive session.")
                break
            if not question:
                continue
            if question.lower() in ('exit', 'quit'):
                print("Exiting interactive session.")
                break
            # Use the agent to answer; it will prefer Bedrock if enabled
            try:
                answer = agent.answer_question(question, data, result_csv)
                print("\nAnswer:\n" + str(answer) + "\n")
            except Exception as e:
                print(f"Error while answering question: {e}")


if __name__ == '__main__':
    main()