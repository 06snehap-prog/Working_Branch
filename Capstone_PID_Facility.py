import csv
import json
from datetime import datetime, timedelta

# Load existing patients to maintain referential integrity
with open("patients.json", "r", encoding="utf-8") as f:
    patients = json.load(f)

# 1. Facility Directory Dataset
facilities = [
    {"facility_id": "SYN-PHC-01", "facility_name": "Sitamarhi Primary Health Centre", "type": "PHC", "state": "Bihar", "district": "Sitamarhi"},
    {"facility_id": "SYN-CHC-02", "facility_name": "Gorakhpur Community Health Centre", "type": "CHC", "state": "Uttar Pradesh", "district": "Gorakhpur"},
    {"facility_id": "SYN-DH-03", "facility_name": "Lucknow District Hospital", "type": "DH", "state": "Uttar Pradesh", "district": "Lucknow"}
]

with open("facility_directory.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["facility_id", "facility_name", "type", "state", "district"])
    writer.writeheader()
    writer.writerows(facilities)

# 2. AI Triage Audit Log Dataset (Simulating model outputs vs Ground Truth)
audit_logs = []
for idx, p in enumerate(patients):
    gt = p["ground_truth"]
    # Simulate a slight variance or match for demonstration
    audit_logs.append({
        "audit_id": f"LOG-90{idx+1:02d}",
        "patient_id": p["patient_id"],
        "inference_timestamp": "2026-11-11 10:30:00",
        "model_version": "us.amazon.nova-micro-v1:0",
        "predicted_priority": gt["expected_priority"],
        "actual_red_flags_count": len(gt["expected_red_flags"]),
        "matched_ground_truth": True,
        "token_usage": 342 + (idx * 15),
        "latency_ms": 412
    })

with open("audit_ai_triage_logs.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=audit_logs[0].keys())
    writer.writeheader()
    writer.writerows(audit_logs)

# 3. Appointments & Follow-up Tracker Dataset
appointments = []
for p in patients:
    if p["ground_truth"]["expected_priority"] == "high":
        status = "Overdue"
        appt_date = "2026-10-25"
    elif p["ground_truth"]["expected_priority"] == "medium":
        status = "Scheduled"
        appt_date = "2026-11-20"
    else:
        status = "Completed"
        appt_date = "2026-09-15"
        
    appointments.append({
        "appointment_id": f"APT-10{p['patient_id'].split('-')[1]}",
        "patient_id": p["patient_id"],
        "facility_id": p["encounters"][-1]["facility"] if p["encounters"] else "SYN-PHC-01",
        "scheduled_date": appt_date,
        "status": status,
        "notes": f"Follow-up for {p['conditions'][0]['name'] if p['conditions'] else 'Routine Checkup'}"
    })

with open("appointments_schedule.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=appointments[0].keys())
    writer.writeheader()
    writer.writerows(appointments)

print("Successfully generated: facility_directory.csv, audit_ai_triage_logs.csv, appointments_schedule.csv")