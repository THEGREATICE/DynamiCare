import json
import requests

API_KEY = "BioPortal API key"  # BioPortal API key
INPUT_FILE = "agentclinic_medqa_extended.jsonl"
OUTPUT_FILE = "converted_cases.jsonl"

def get_first_icd9_code(term, api_key):
    url = "http://data.bioontology.org/search"
    params = {
        "q": term,
        "ontologies": "ICD9CM",
        "apikey": api_key
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        for result in data.get("collection", []):
            code_url = result.get("@id", "")
            if code_url and "ICD9CM" in code_url:
                return code_url.split("/")[-1]
    except Exception as e:
        print(f"API error for '{term}': {e}")
    return None

with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    kept, skipped = 0, 0
    for line in infile:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            exam = record.get("OSCE_Examination", {})
            diagnosis_text = exam.get("Correct_Diagnosis", "")
            icd9_code = get_first_icd9_code(diagnosis_text, API_KEY)
            if icd9_code:
                exam["Correct_Diagnosis"] = icd9_code
                record["OSCE_Examination"] = exam
                outfile.write(json.dumps(record) + "\n")
                kept += 1
                print(f"✓ {diagnosis_text} → {icd9_code}")
            else:
                skipped += 1
                print(f"✗ Skipped: {diagnosis_text}")
        except Exception as e:
            skipped += 1
            print(f"✗ Error: {e}")

print(f"\n Done yay: {kept} cases kept, {skipped} skipped.")
