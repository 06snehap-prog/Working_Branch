"""SwasthSaar mock dataset generator.
ALL DATA IS SYNTHETIC. Names, places and values are invented for a course project.
Reference date for the dataset: 2026-11-11.
Codes (ICD-10, LOINC) are indicative; verify against the ABDM IG before claiming conformance.
"""
import csv
import json

REF = "2026-11-11"

LOINC = {
    "BP systolic": ("8480-6", "mmHg"),
    "BP diastolic": ("8462-4", "mmHg"),
    "HbA1c": ("4548-4", "%"),
    "Fasting glucose": ("1558-6", "mg/dL"),
    "Hemoglobin": ("718-7", "g/dL"),
    "Weight": ("29463-7", "kg"),
    "Creatinine": ("2160-0", "mg/dL"),
    "eGFR": ("62238-1", "mL/min/1.73m2"),
    "LDL cholesterol": ("13457-7", "mg/dL"),
    "Potassium": ("2823-3", "mmol/L"),
    "Urine protein (dipstick)": ("", "grade"),
}

ALL_SCOPE = ["OPConsultRecord", "PrescriptionRecord", "DiagnosticReportRecord",
             "WellnessRecord", "ImmunizationRecord"]


def ob(d, n, v):
    code, unit = LOINC[n]
    return {"date": d, "name": n, "loinc": code, "value": v, "unit": unit}


def bp(d, s, di):
    return [ob(d, "BP systolic", s), ob(d, "BP diastolic", di)]


def enc(d, fac, typ, reason, note=""):
    return {"date": d, "facility": fac, "type": typ, "reason": reason, "note": note}


def cond(icd, name, since):
    return {"icd10": icd, "name": name, "since": since}


def med(drug, dose, start, last_fill, fac, stop=None):
    return {"drug": drug, "dose": dose, "start": start, "last_fill": last_fill,
            "facility": fac, "stopped": stop}


def P(pid, name, age, sex, state, district, lang, **kw):
    consent = kw.get("consent") or {
        "status": "granted", "purpose": "care-consultation",
        "scope": ALL_SCOPE, "valid_until": "2027-05-01"}
    return {
        "patient_id": pid,
        "abha_address": f"{pid.lower().replace('-', '')}@synthetic",
        "demographics": {"name": name, "age": age, "sex": sex, "state": state,
                         "district": district, "preferred_language": lang},
        "consent": consent,
        "encounters": kw.get("encounters", []),
        "conditions": kw.get("conditions", []),
        "observations": kw.get("observations", []),
        "medications": kw.get("medications", []),
        "allergies": kw.get("allergies", []),
        "immunizations": kw.get("immunizations", []),
        "data_quality": {"missing": kw.get("missing", []),
                         "conflicts": kw.get("conflicts", [])},
        "ground_truth": {"expected_priority": kw["priority"],
                         "expected_red_flags": kw.get("flags", []),
                         "expected_summary": kw["summary"]},
    }


F1, F2, F3 = "SYN-PHC-01", "SYN-CHC-02", "SYN-DH-03"

patients = [
    P("SYN-001", "Sunita Devi", 54, "F", "Bihar", "Sitamarhi", "Hindi",
      encounters=[enc("2025-10-05", F1, "OP", "Diabetes follow-up"),
                  enc("2026-01-15", F1, "OP", "Diabetes follow-up"),
                  enc("2026-04-10", F1, "OP", "Diabetes and BP review", "Follow-up advised in 3 months")],
      conditions=[cond("E11", "Type 2 diabetes", "2019"), cond("I10", "Hypertension", "2021")],
      observations=[ob("2025-10-05", "HbA1c", 7.6), ob("2026-01-15", "HbA1c", 8.8),
                    ob("2026-04-10", "HbA1c", 9.8), *bp("2026-04-10", 152, 94)],
      medications=[med("Metformin", "500 mg twice daily", "2019-06-01", "2026-04-10", F1),
                   med("Glimepiride", "1 mg once daily", "2022-03-01", "2026-04-10", F1)],
      priority="high",
      flags=["HbA1c rising 7.6 -> 8.8 -> 9.8", "Follow-up overdue (due 2026-07-10)", "BP above target at last visit"],
      summary="54F, T2DM and hypertension. HbA1c worsening over 6 months, last seen 2026-04-10 and 3-month follow-up missed."),
    P("SYN-002", "Ramesh Yadav", 61, "M", "Uttar Pradesh", "Gorakhpur", "Hindi",
      encounters=[enc("2026-03-02", F1, "OP", "BP check"), enc("2026-06-02", F1, "OP", "Headache, BP check")],
      conditions=[cond("I10", "Hypertension", "2018")],
      observations=[*bp("2026-03-02", 164, 100), *bp("2026-06-02", 168, 102)],
      medications=[med("Amlodipine", "5 mg once daily", "2018-05-01", "2026-03-02", F1)],
      priority="high",
      flags=["BP uncontrolled on two consecutive visits", "No medication refill recorded for over 7 months (possible lapse)"],
      summary="61M, hypertension on amlodipine 5 mg. BP 168/102 at last visit and no refill since March."),
    P("SYN-003", "Lakshmi Kannan", 26, "F", "Tamil Nadu", "Salem", "Tamil",
      encounters=[enc("2026-09-15", F1, "ANC", "Antenatal visit"), enc("2026-10-20", F1, "ANC", "Antenatal visit", "Approx 29 weeks")],
      conditions=[cond("Z34", "Pregnancy, supervised (LMP 2026-04-01)", "2026-04")],
      observations=[*bp("2026-09-15", 128, 82), ob("2026-09-15", "Hemoglobin", 10.2),
                    *bp("2026-10-20", 148, 96), ob("2026-10-20", "Hemoglobin", 8.9),
                    ob("2026-10-20", "Urine protein (dipstick)", "1+")],
      medications=[med("Iron-folic acid", "1 tablet daily", "2026-05-01", "2026-10-20", F1)],
      priority="high",
      flags=["BP 148/96 in third trimester (rising from 128/82)", "New proteinuria 1+", "Anemia Hb 8.9 g/dL"],
      summary="26F, about 32 weeks pregnant. BP and urine protein rising, Hb 8.9: possible pre-eclampsia risk, needs doctor assessment."),
    P("SYN-004", "Mohan Lal", 67, "M", "Rajasthan", "Alwar", "Hindi",
      encounters=[enc("2025-08-20", F2, "OP", "Annual review"), enc("2026-08-14", F2, "OP", "Annual review")],
      conditions=[cond("E11", "Type 2 diabetes", "2012"), cond("I10", "Hypertension", "2014")],
      observations=[ob("2025-08-20", "eGFR", 52), ob("2026-08-14", "eGFR", 38),
                    ob("2026-08-14", "Creatinine", 1.9), ob("2026-08-14", "HbA1c", 7.9), *bp("2026-08-14", 142, 88)],
      medications=[med("Metformin", "1000 mg twice daily", "2013-01-01", "2026-08-14", F2),
                   med("Telmisartan", "40 mg once daily", "2016-01-01", "2026-08-14", F2)],
      priority="high",
      flags=["eGFR fell 52 -> 38 in a year", "Metformin dose may need review at eGFR below 45 (doctor to decide)"],
      summary="67M, diabetes and hypertension. Kidney function declining; current metformin dose should be reviewed by the doctor."),
    P("SYN-005", "Fatima Begum", 34, "F", "West Bengal", "Murshidabad", "Bengali",
      encounters=[enc("2022-07-09", F1, "OP", "Skin rash after antibiotic"), enc("2025-12-03", F1, "OP", "Sore throat")],
      allergies=[{"substance": "Penicillin", "reaction": "Skin rash", "recorded": "2022-07-09"}],
      observations=[*bp("2025-12-03", 112, 72)],
      missing=["No chronic conditions recorded"],
      priority="medium",
      flags=["Documented penicillin allergy"],
      summary="34F, no chronic illness. Penicillin allergy (rash) must be visible before any antibiotic is chosen."),
    P("SYN-006", "Kamala Bai", 74, "F", "Karnataka", "Raichur", "Kannada",
      encounters=[enc("2026-05-12", F2, "OP", "Fall at home"), enc("2026-09-22", F2, "OP", "Second fall, knee pain")],
      conditions=[cond("I10", "Hypertension", "2010"), cond("E11", "Type 2 diabetes", "2015"),
                  cond("E03", "Hypothyroidism", "2018"), cond("M17", "Knee osteoarthritis", "2020")],
      observations=[*bp("2026-09-22", 134, 80), ob("2026-09-22", "HbA1c", 6.4)],
      medications=[med("Amlodipine", "5 mg", "2010-01-01", "2026-09-22", F2),
                   med("Metformin", "500 mg twice daily", "2015-01-01", "2026-09-22", F2),
                   med("Glimepiride", "2 mg", "2017-01-01", "2026-09-22", F2),
                   med("Levothyroxine", "50 mcg", "2018-01-01", "2026-09-22", F2),
                   med("Aspirin", "75 mg", "2019-01-01", "2026-09-22", F2),
                   med("Pantoprazole", "40 mg", "2024-01-01", "2026-09-22", F2),
                   med("Diclofenac", "50 mg as needed", "2026-05-12", "2026-09-22", F2)],
      priority="high",
      flags=["Two falls in 4 months", "7 active medications (polypharmacy)",
             "NSAID in age 74 with hypertension and diabetes",
             "Sulfonylurea with HbA1c 6.4 in an elderly patient (hypoglycaemia risk)"],
      summary="74F, four chronic conditions and 7 medicines, two recent falls. Medicine review suggested (NSAID, sulfonylurea)."),
    P("SYN-007", "Aarav Patil", 3, "M", "Maharashtra", "Nanded", "Marathi",
      encounters=[enc("2026-09-30", F1, "OP", "Cough and cold")],
      observations=[ob("2026-09-30", "Weight", 11.0)],
      immunizations=[{"vaccine": "BCG", "date": "2023-11-05"}, {"vaccine": "OPV", "date": "2023-11-05"},
                     {"vaccine": "Pentavalent 1-3", "date": "2024-02-10"}, {"vaccine": "MR dose 1", "date": "2024-08-12"}],
      priority="medium",
      flags=["Booster doses (MR 2 / DPT booster) overdue for age 3", "Low weight for age, needs growth chart check"],
      summary="3-year-old boy, booster vaccines not recorded and weight 11.0 kg. Check immunization and growth."),
    P("SYN-008", "Suresh Kumar", 45, "M", "Uttar Pradesh", "Kanpur", "Hindi",
      encounters=[enc("2021-06-14", F1, "OP", "Fever")],
      medications=[med("Paracetamol", "500 mg", "2021-06-14", "2021-06-14", F1, stop="2021-06-19")],
      missing=["Vitals", "Lab results", "Conditions", "Allergies", "Any record after 2021"],
      priority="low",
      flags=[],
      summary="Only one old fever visit (2021). Not enough data to summarize: system must say so and not infer anything."),
    P("SYN-009", "Anita Sharma", 43, "F", "Madhya Pradesh", "Indore", "Hindi",
      encounters=[enc("2026-08-01", F1, "OP", "BP check"), enc("2026-08-14", F2, "OP", "BP check")],
      conditions=[cond("I10", "Hypertension", "2024")],
      observations=[*bp("2026-08-01", 138, 88), ob("2026-08-01", "Weight", 62),
                    ob("2026-08-14", "Weight", 72)],
      conflicts=["Name spelled Anita Sharma (PHC-01) vs Anita Sarma (CHC-02)",
                 "Year of birth 1983 vs 1981 across facilities",
                 "Weight 62 kg vs 72 kg within 13 days"],
      priority="low",
      flags=["Record identity and weight conflict across facilities"],
      summary="43F with hypertension. Two facilities disagree on name spelling, birth year and weight: show the conflict, do not pick silently."),
    P("SYN-010", "Rohit Singh", 29, "M", "Punjab", "Ludhiana", "Punjabi",
      encounters=[enc("2026-10-10", F1, "OP", "Routine check-up")],
      observations=[*bp("2026-10-10", 118, 76), ob("2026-10-10", "Weight", 70)],
      priority="low", flags=[],
      summary="29M, healthy routine visit with normal vitals. No concerns (negative control)."),
    P("SYN-011", "Vijay Patil", 52, "M", "Maharashtra", "Satara", "Marathi",
      encounters=[enc("2026-10-02", F1, "OP", "Chest tightness on climbing stairs", "Smoker, 20 years")],
      conditions=[cond("I10", "Hypertension", "2023")],
      observations=[*bp("2026-10-02", 152, 94), ob("2026-10-02", "LDL cholesterol", 168),
                    ob("2026-10-02", "Fasting glucose", 118)],
      missing=["No ECG on record"],
      priority="high",
      flags=["Exertional chest symptoms with several cardiac risk factors", "No ECG recorded", "LDL 168 and BP 152/94"],
      summary="52M smoker, hypertension, recent exertional chest tightness and no ECG on file. Needs urgent clinical evaluation."),
    P("SYN-012", "Ganesh Nayak", 38, "M", "Odisha", "Koraput", "Odia",
      encounters=[enc("2026-08-25", F2, "OP", "Pulmonary TB diagnosed"), enc("2026-09-24", F2, "OP", "Drug refill")],
      conditions=[cond("A15", "Pulmonary tuberculosis", "2026-08")],
      medications=[med("Anti-TB regimen (fixed-dose)", "per weight band", "2026-08-25", "2026-09-24", F2)],
      priority="high",
      flags=["Treatment possibly interrupted: last refill 2026-09-24", "2-month follow-up (due about 2026-10-25) not recorded"],
      summary="38M on TB treatment since 2026-08-25. No refill or follow-up after 2026-09-24: risk of treatment interruption."),
    P("SYN-013", "Pooja Verma", 22, "F", "Uttar Pradesh", "Lucknow", "Hindi",
      encounters=[enc("2026-08-14", F3, "ER", "Wheeze and breathlessness"), enc("2026-10-30", F3, "ER", "Wheeze and breathlessness")],
      conditions=[cond("J45", "Asthma", "2019")],
      medications=[med("Salbutamol inhaler", "as needed", "2019-01-01", "2026-10-30", F3)],
      priority="medium",
      flags=["Two emergency visits in 3 months", "Reliever only, no controller therapy recorded"],
      summary="22F with asthma, two ER visits in 11 weeks and only a reliever inhaler on record. Review asthma control."),
    P("SYN-014", "Shanti Oraon", 29, "F", "Jharkhand", "Gumla", "Hindi",
      encounters=[enc("2026-10-18", F1, "ANC", "Antenatal visit", "About 28 weeks; diet advice given; review in 1 week")],
      conditions=[cond("O24.4", "Gestational diabetes", "2026-10")],
      observations=[ob("2026-10-18", "Fasting glucose", 108), *bp("2026-10-18", 118, 76)],
      priority="high",
      flags=["Gestational diabetes follow-up overdue (due about 2026-10-25)"],
      summary="29F, about 32 weeks pregnant with gestational diabetes (fasting glucose 108). Planned review not recorded."),
    P("SYN-015", "Geeta Rao", 49, "F", "Telangana", "Warangal", "Telugu",
      encounters=[enc("2026-06-01", F1, "OP", "Diabetes follow-up"), enc("2026-09-01", F1, "OP", "Diabetes follow-up", "Next visit December")],
      conditions=[cond("E11", "Type 2 diabetes", "2020")],
      observations=[ob("2026-06-01", "HbA1c", 6.8), ob("2026-09-01", "HbA1c", 6.6), *bp("2026-09-01", 124, 80)],
      medications=[med("Metformin", "500 mg twice daily", "2020-03-01", "2026-09-01", F1)],
      priority="low", flags=[],
      summary="49F, diabetes well controlled (HbA1c 6.6) and visits on schedule (negative control for chronic care)."),
    P("SYN-016", "Harpal Singh", 58, "M", "Punjab", "Amritsar", "Punjabi",
      encounters=[enc("2026-07-05", F1, "OP", "BP review"), enc("2026-10-05", F1, "OP", "BP review")],
      conditions=[cond("I10", "Hypertension", "2019")],
      observations=[*bp("2026-07-05", 130, 82), *bp("2026-10-05", 126, 78)],
      medications=[med("Amlodipine", "5 mg once daily", "2019-04-01", "2026-10-05", F1)],
      priority="low", flags=[],
      summary="58M, hypertension controlled with regular refills (negative control)."),
    P("SYN-017", "Meena Kumari", 16, "F", "Bihar", "Gaya", "Magahi/Hindi",
      encounters=[enc("2026-09-10", F1, "OP", "Fatigue and dizziness")],
      observations=[ob("2026-09-10", "Hemoglobin", 8.1), ob("2026-09-10", "Weight", 41)],
      missing=["No iron supplementation recorded"],
      priority="medium",
      flags=["Moderate anemia Hb 8.1 g/dL", "No iron-folic acid treatment documented"],
      summary="16F with fatigue and Hb 8.1; no treatment on record. Anemia work-up and supplementation to be considered."),
    P("SYN-018", "Bhola Prasad", 63, "M", "Bihar", "Muzaffarpur", "Hindi",
      encounters=[enc("2026-03-18", F2, "OP", "Foot ulcer, left"), enc("2026-09-12", F2, "OP", "Foot ulcer, left, recurrent")],
      conditions=[cond("E11", "Type 2 diabetes", "2010")],
      observations=[ob("2026-03-18", "HbA1c", 9.9), ob("2026-09-12", "HbA1c", 10.4)],
      medications=[med("Metformin", "500 mg twice daily", "2010-01-01", "2026-09-12", F2)],
      priority="high",
      flags=["Recurrent diabetic foot ulcer", "HbA1c 10.4 on metformin alone"],
      summary="63M, long-standing diabetes with recurrent foot ulcer and HbA1c 10.4. High risk, needs referral discussion."),
    P("SYN-019", "Neha Joshi", 41, "F", "Uttarakhand", "Almora", "Hindi",
      consent={"status": "expired", "purpose": "care-consultation",
               "scope": ["PrescriptionRecord"], "valid_until": "2026-09-30"},
      encounters=[enc("2026-02-11", F1, "OP", "Diabetes review")],
      conditions=[cond("E11", "Type 2 diabetes", "2021")],
      observations=[ob("2026-02-11", "HbA1c", 8.2)],
      priority="n/a (consent required)",
      flags=[],
      summary="Consent expired on 2026-09-30 and covered prescriptions only. The system must NOT summarize clinical data; it must ask for fresh patient consent."),
    P("SYN-020", "Salim Khan", 59, "M", "Delhi", "North Delhi", "Hindi/Urdu",
      encounters=[enc("2026-08-05", F1, "OP", "BP review"), enc("2026-09-20", F2, "OP", "BP review")],
      conditions=[cond("I10", "Hypertension", "2020")],
      observations=[*bp("2026-09-20", 132, 84), ob("2026-09-20", "Potassium", 5.4)],
      medications=[med("Ramipril", "5 mg once daily", "2025-01-10", "2026-08-05", F1),
                   med("Losartan", "50 mg once daily", "2026-09-20", "2026-09-20", F2)],
      conflicts=["Two facilities prescribed different blood-pressure drug classes"],
      priority="high",
      flags=["ACE inhibitor (ramipril) and ARB (losartan) both active", "Potassium 5.4 mmol/L (high)"],
      summary="59M, hypertension. Two facilities prescribed an ACE inhibitor and an ARB, and potassium is 5.4. Needs medication reconciliation."),
]

with open("patients.json", "w", encoding="utf-8") as f:
    json.dump(patients, f, indent=2, ensure_ascii=False)

summary_cols = ["patient_id", "name", "age", "sex", "state", "language", "conditions",
                "n_encounters", "last_encounter", "n_observations", "n_medications",
                "allergies", "consent_status", "missing_fields", "conflicts",
                "expected_priority", "n_red_flags", "expected_red_flags"]
with open("patients_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(summary_cols)
    for p in patients:
        d = p["demographics"]
        last = max((e["date"] for e in p["encounters"]), default="")
        w.writerow([p["patient_id"], d["name"], d["age"], d["sex"], d["state"],
                    d["preferred_language"],
                    "; ".join(c["name"] for c in p["conditions"]),
                    len(p["encounters"]), last, len(p["observations"]),
                    len(p["medications"]),
                    "; ".join(a["substance"] for a in p["allergies"]),
                    p["consent"]["status"],
                    "; ".join(p["data_quality"]["missing"]),
                    "; ".join(p["data_quality"]["conflicts"]),
                    p["ground_truth"]["expected_priority"],
                    len(p["ground_truth"]["expected_red_flags"]),
                    " | ".join(p["ground_truth"]["expected_red_flags"])])

with open("observations_long.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["patient_id", "date", "name", "loinc", "value", "unit"])
    for p in patients:
        for o in p["observations"]:
            w.writerow([p["patient_id"], o["date"], o["name"], o["loinc"], o["value"], o["unit"]])

print("patients:", len(patients))
