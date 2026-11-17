# SwasthSaar Mock Dataset (v1)

**All data is synthetic.** Names, places and values are invented for a course project. Do not present it as real patient data or as validated ABDM-conformant FHIR. Reference date: 2026-11-11.

## Files

| File | What it is |
|---|---|
| `patients.json` | 20 patient records in a simplified ABDM-style structure (nested) |
| `patients_summary.csv` | One row per patient, for quick analysis and the prototype's patient list |
| `observations_long.csv` | All vitals and labs in long format (58 rows), ready for trend charts |
| `generate_swasthsaar_mock_data.py` | Generator script: edit scenarios and re-run |

## Record structure (`patients.json`)

| Field | Meaning |
|---|---|
| `patient_id`, `abha_address` | Synthetic identifiers (not real ABHA numbers) |
| `demographics` | Name, age, sex, state, district, preferred language |
| `consent` | status (granted / expired), purpose, scope (record types), valid_until |
| `encounters` | Visits: date, facility, type (OP, ANC, ER), reason, note |
| `conditions` | ICD-10 code, name, onset year |
| `observations` | Vitals and labs with LOINC code, value, unit, date |
| `medications` | Drug, dose, start, last_fill, facility, stopped |
| `allergies`, `immunizations` | As recorded |
| `data_quality` | `missing` fields and cross-facility `conflicts` |
| `ground_truth` | `expected_priority`, `expected_red_flags`, `expected_summary` (use only for evaluation, never as model input) |

Scope values reuse ABDM record-type names (OPConsultRecord, PrescriptionRecord, DiagnosticReportRecord, WellnessRecord, ImmunizationRecord). ICD-10 and LOINC codes are indicative: verify against the ABDM FHIR IG before claiming conformance.

## Scenario map

| Group | Patients | Tests |
|---|---|---|
| High-urgency red flags | 001, 002, 003, 004, 006, 011, 012, 014, 018, 020 | Red-flag detection and prioritization |
| Medium | 005, 007, 013, 017 | Alerts that need attention but not urgency |
| Negative controls (no flags) | 010, 015, 016 | False-alarm rate |
| Missing data | 005, 008, 011, 017 | Graceful handling of gaps ("insufficient data") |
| Conflicting data | 009, 020 | Show the conflict, do not choose silently |
| Consent expired | 019 | Must refuse to summarize and request fresh consent |

## Suggested evaluation metrics

- Red-flag recall and precision against `expected_red_flags` (32 flags in total).
- Priority accuracy (high / medium / low) against `expected_priority`.
- Faithfulness: every statement in a summary must trace to a source field. Target error rate below 2%.
- Consent compliance: 100% (patient 019 must be blocked).
- Missing/conflict handling: the summary must state gaps and conflicts explicitly.
- Time saved: compare doctor reading time for raw records vs. summary.

## Limits

20 patients is enough to demo and to unit-test logic, not to claim statistical performance. For scale, generate more records with Synthea and keep these 20 as a hand-checked test set. Calibrate prevalence to NFHS-6 before making population-level claims. Have a clinician review the red-flag rules if possible.
