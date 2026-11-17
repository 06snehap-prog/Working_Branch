import os
import json
from supabase import create_client

url = "https://celmwyrtewqfgjlcdiqf.supabase.co"
key = "sb_publishable_SJCtTyiuxWFcD0qj_7Ck1A_3vCXP9H0"  # Replace with your actual Supabase Anon/Service Role Key

supabase = create_client(url, key)

with open("patients.json", "r", encoding="utf-8") as f:
    patients = json.load(f)

for p in patients:
    pid = p["patient_id"]
    d = p["demographics"]
    
    # 1. Upsert Patient
    supabase.table("patients").upsert({
        "patient_id": pid,
        "abha_address": p["abha_address"],
        "name": d["name"],
        "age": d["age"],
        "sex": d["sex"],
        "state": d["state"],
        "district": d["district"],
        "preferred_language": d["preferred_language"]
    }).execute()

    # 2. Insert Consent
    c = p["consent"]
    supabase.table("patient_consents").insert({
        "patient_id": pid,
        "status": c["status"],
        "purpose": c["purpose"],
        "scope": c["scope"],
        "valid_until": c["valid_until"]
    }).execute()

    # 3. Insert Encounters
    for e in p["encounters"]:
        supabase.table("encounters").insert({
            "patient_id": pid, "encounter_date": e["date"], "facility": e["facility"],
            "encounter_type": e["type"], "reason": e["reason"], "note": e["note"]
        }).execute()

    # 4. Insert Conditions
    for cond in p["conditions"]:
        supabase.table("conditions").insert({
            "patient_id": pid, "icd10": cond["icd10"],
            "condition_name": cond["name"], "since_date": cond["since"]
        }).execute()

    # 5. Insert Observations
    for obs in p["observations"]:
        supabase.table("observations").insert({
            "patient_id": pid, "observation_date": obs["date"], "observation_name": obs["name"],
            "loinc": obs["loinc"], "value": str(obs["value"]), "unit": obs["unit"]
        }).execute()

    # 6. Insert Medications
    for m in p["medications"]:
        supabase.table("medications").insert({
            "patient_id": pid, "drug": m["drug"], "dose": m["dose"],
            "start_date": m["start"], "last_fill_date": m["last_fill"],
            "facility": m["facility"], "stopped_date": m["stopped"]
        }).execute()

    # 7. Insert Allergies
    for a in p["allergies"]:
        supabase.table("allergies").insert({
            "patient_id": pid, "substance": a["substance"],
            "reaction": a["reaction"], "recorded_date": a["recorded"]
        }).execute()

    # 8. Insert Immunizations
    for imm in p["immunizations"]:
        supabase.table("immunizations").insert({
            "patient_id": pid, "vaccine": imm["vaccine"], "immunization_date": imm["date"]
        }).execute()

    # 9. Data Quality & Ground Truth
    dq = p["data_quality"]
    supabase.table("data_quality_flags").insert({
        "patient_id": pid, "missing_fields": dq["missing"], "conflicts": dq["conflicts"]
    }).execute()

    gt = p["ground_truth"]
    supabase.table("ground_truth").insert({
        "patient_id": pid, "expected_priority": gt["expected_priority"],
        "expected_red_flags": gt["expected_red_flags"], "expected_summary": gt["expected_summary"]
    }).execute()

print("All 20 patient records seeded successfully into Supabase!")