"""
### Answer agent
1. Keyword Extraction
    - parse out medical concepts, demographics, time references.

2. Rule-Based Section Matching
    - Try to map keywords to JSON sections using: Keyword-to-section dictionaries, Simple ontologies (e.g., UMLS terms)

3. Data Retrieval 
    - If a section is matched, extract the relevant information directly.

4. Fallback to LLM Inference (if no section match or ambiguous)
    - If: No sections were confidently matched or Retrieved data is empty or uncertain
    - Then: Use the full patient JSON or a reduced, structured summary as input and ask the LLM directly
"""

import json
import re
import time
from openai import OpenAI
from datetime import datetime
import argparse
with open("config.json") as f:
    config = json.load(f)
client = OpenAI(api_key=config["OPENAI_API_KEY"])

# Step 1: Extract keywords
def extract_keywords(question):
    keywords = re.findall(r'\b\w+\b', question.lower())
    return keywords

# Step 2: Map keywords to section
keyword_mapping = {
    # Demographics
    ('age',): ('Demographics', 'age'),
    ('language', 'english', 'spanish'): ('Demographics', 'language'),
    ('religion', 'religious'): ('Demographics', 'religion'),
    ('marital', 'married', 'single', 'divorced', 'widowed'): ('Demographics', 'marital_status'),
    ('gender', 'sex'): ('Demographics', 'gender'),
    ('insurance',): ('Demographics', 'insurance'),
    ('ethnicity',): ('Demographics', 'ethnicity'),

    # Medications
    ('admission medications', 'initial meds', 'medications', 'medication', 'drugs'): [('Medications on Admission', None),('Medications', None)],

    # Procedures
    ('procedure', 'surgery', 'operation'): [('Procedure', None), ('Major Surgical or Invasive Procedure', None)],

    # Imaging and reports
    ('electrocardiogram', 'ecg'): ('ECG', None),
    ('echocardiogram', 'echo'): ('Echo', None),
    ('radiology', 'x-ray', 'ct', 'mri', 'imaging'): ('Radiology', None),

    # History
    ('hpi', 'present illness', 'history of present illness'): ('History of Present Illness', None),
    ('past medical', 'pmh', 'past medical history'): ('Past Medical History', None),
    ('family history',): ('Family History', None),
    ('social history', 'drinking', 'smoking', 'drug use', 'tobacco', 'alcohol'): ('Social History', None),

    # Allergies
    ('allergy', 'allergies', 'allergic'): ('Allergies', None),

    # Physical exam (with HEENT included)
    ('heent', 'head', 'eyes', 'ears', 'nose', 'throat'): ('Physical Exam.Admission', 'HEENT'),
    ('physical exam',): ('Physical Exam.Admission', None),



# === High-Level Chart Sections ===
    ('vital signs', 'bp', 'blood pressure', 'heart rate', 'temperature', 'ectopy', 'vitals'): ('Routine Vital Signs', None),
    ('respiratory', 'ventilator', 'peep', 'tidal volume', 'o2 sat', 'oxygen'): ('Respiratory', None),
    ('labs', 'blood test', 'cbc', 'lab results'): [
        ('Hematology', None),
        ('Chemistry', None),
        ('Blood Gas', None)
    ],
    ('weight', 'height', 'code status', 'precautions', 'daily weight'): ('General', None),
    ('urine', 'stool', 'bowel', 'diet', 'gi', 'gu'): ('GI/GU', None),
    ('extremities', 'perfusion', 'angio', 'limb temperature'): ('Cardiovascular', None),

    # === Direct Lab Label Matches (samples) ===
    ('hemoglobin',): ('Hematology', 'Hemoglobin'),
    ('hematocrit',): ('Hematology', 'Hematocrit'),
    ('wbc', 'white blood cell'): ('Hematology', 'WBC'),
    ('ptt',): ('Hematology', 'PTT'),
    ('inr',): ('Hematology', 'INR(PT)'),
    ('sodium',): ('Chemistry', 'Sodium'),
    ('potassium',): ('Chemistry', 'Potassium'),
    ('magnesium',): ('Chemistry', 'Magnesium'),
    ('glucose',): ('Chemistry', 'Glucose'),
    ('creatinine',): ('Chemistry', 'Creatinine'),
    ('anion gap',): ('Chemistry', 'Anion Gap'),
    ('phosphorous',): ('Chemistry', 'Phosphate'),
    ('bun',): ('Chemistry', 'Urea Nitrogen'),
    ('bicarbonate', 'hco3'): ('Chemistry', 'Bicarbonate'),
    ('ph arterial', 'ph', 'base excess'): ('Blood Gas', 'PH'),
    ('po2', 'oxygen pressure'): ('Blood Gas', 'pO2'),
    ('pco2', 'carbon dioxide pressure'): ('Blood Gas', 'pCO2'),
    ('o2 saturation', 'oxygen saturation'): ('Blood Gas', 'Oxygen Saturation'),
    ('lactate',): ('Blood Gas', 'Lactate'),

    # === Chartdata Direct Matches (examples from previous message) ===
    ('heart rate',): ('Routine Vital Signs', 'Heart Rate'),
    ('temperature',): ('Routine Vital Signs', 'Temperature Fahrenheit'),
    ('blood pressure',): ('Routine Vital Signs', 'Non Invasive Blood Pressure systolic'),
    ('respiratory rate',): ('Respiratory', 'Respiratory Rate'),
    ('o2 flow',): ('Respiratory', 'O2 Flow'),
    ('peep',): ('Respiratory', 'PEEP set'),
    ('tidal volume',): ('Respiratory', 'Tidal Volume (observed)'),
    ('weight',): ('General', 'Daily Weight'),
    ('height',): ('General', 'Height (cm)'),
    ('urine output',): ('GI/GU', 'Urine Source'),
    ('stool consistency',): ('GI/GU', 'Stool Consistency'),
    ('bowel sounds',): ('GI/GU', 'Bowel Sounds'),
    ('extremities',): ('Cardiovascular', 'RUE Temp'),

    # === Broad Mappings for Catch-All Matching ===
    ('chemistry',): ('Chemistry', None),
    ('hematology',): ('Hematology', None),
    ('blood gas', 'arterial gas'): ('Blood Gas', None),
}


def match_keywords_to_sections(question, keyword_mapping):
    question_text = question.lower()
    matched_sections = set()

    # Main keyword mapping match
    for key_group, targets in keyword_mapping.items():
        for keyword in key_group:
            if keyword in question_text:
                if isinstance(targets, list):
                    matched_sections.update(targets)
                else:
                    matched_sections.add(targets)

    # Fallback: direct section name match
    if not matched_sections:
        section_names = set()
        for targets in keyword_mapping.values():
            if isinstance(targets, list):
                section_names.update(t[0] for t in targets)
            else:
                section_names.add(targets[0])

        for section in section_names:
            if section.lower() in question_text:
                matched_sections.add((section, None))

    return list(matched_sections)

# Step 3: Extract relevant info
# has time: ECG, Echo, Radiology, chart, lab
time_sensitive_sections = {
    "Routine Vital Signs", "Labs", "Respiratory", "Cardiovascular", "General", "GI/GU",
    "Hematology", "Blood Gas", "Chemistry", "ECG", "Echo", "Radiology"
}

def has_time_reference(question):
    # Simple check: YYYY-MM-DD or phrases like "on March 1st"
    date_pattern = r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b"
    relative_time_keywords = ['yesterday', 'today', 'tomorrow', 'last', 'next', 'previous', 'week', 'month', 'year', 'day']
    return re.search(date_pattern, question) or any(word in question.lower() for word in relative_time_keywords)

def extract_date_from_question(question):
    date_match = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", question)
    if date_match:
        try:
            y, m, d = map(int, date_match.groups())
            return datetime(y, m, d).date()
        except:
            return None
    return None

def extract_data(json_data, sections, question=None):
    from datetime import datetime

    patient_data = json_data.get("Patient", {})
    admission_time_str = patient_data.get("Admission_info", {}).get("admission_time", None)
    admission_date = datetime.strptime(admission_time_str, "%Y-%m-%d %H:%M:%S").date() if admission_time_str else None

    target_date = extract_date_from_question(question) if question else None
    results = {}

    time_sensitive_sections = {
        "Routine Vital Signs", "Labs", "Respiratory", "Cardiovascular", "General", "GI/GU",
        "Hematology", "Blood Gas", "Chemistry", "ECG", "Echo", "Radiology"
    }

    for section, field in sections:
        section_data = patient_data.get(section, None)
        if section_data is None:
            results[(section, field)] = None
            continue

        # Section is not time-sensitive or no time filtering needed
        if section not in time_sensitive_sections:
            if field is None:
                results[(section, field)] = section_data
            elif isinstance(section_data, dict):
                results[(section, field)] = section_data.get(field, None)
            else:
                results[(section, field)] = None
            continue

        # --------------------------
        # Handle time-aware sections
        # --------------------------

        # Case 1: section is a list (e.g., ECG, Echo, Radiology)
        if isinstance(section_data, list):
            if target_date:
                filtered = [entry for entry in section_data if isinstance(entry, list) and datetime.strptime(entry[0], "%Y-%m-%d").date() == target_date]
                if not filtered:
                    filtered = [entry for entry in section_data if isinstance(entry, dict) and datetime.strptime(entry.get("time", ""), "%Y-%m-%d").date() == target_date]
                results[(section, field)] = filtered or [section_data[0]] if section_data else None
            else:
                results[(section, field)] = section_data[0] if section_data else None

        # Case 2: section is a dict with time-series per label (e.g., Respiratory["O2 saturation pulseoxymetry"])
        elif isinstance(section_data, dict):
            label_data = {}
            for label, entries in section_data.items():
                if not isinstance(entries, list) or not entries:
                    continue
                if target_date:
                    filtered = [e for e in entries if datetime.strptime(e[0], "%Y-%m-%d %H:%M:%S").date() == target_date]
                    label_data[label] = filtered or [entries[0]]
                else:
                    label_data[label] = [entries[0]]
            results[(section, field)] = label_data

        else:
            results[(section, field)] = section_data

    return results

# Step 4: Generate answer
def generate_answer(question, extracted_data, model):
    # Step 1: Ask GPT if the extracted content is enough
    sufficiency_check = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a QA assistant. Determine if the following content is sufficient to answer the user's question. "
                    "Answer only with 'yes' or 'no'."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Extracted content: {extracted_data}"
                )
            }
        ],
        temperature=0
    )

    answer_ok = sufficiency_check.choices[0].message.content.strip().lower().startswith("yes")

    # Step 2: If content is sufficient, generate an answer using it
    if answer_ok:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical assistant helping simulate a patient in a conversation with a doctor. "
                        "You are given structured health record data (e.g., from a medical chart), and your task is to generate "
                        "a natural, brief, and direct answer to the question based **only on the provided content**. "
                        "Speak as if you are the patient, using first-person phrasing like 'I have...', 'I'm...', or 'My doctor prescribed...'."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Relevant content: {extracted_data}\n\n"
                        "Answer as the patient:"
                    )
                }
            ],
            temperature=0
        )
        return response.choices[0].message.content

    return None  # signal to use fallback

def fallback_llm_answer(question, json_data, model):

    patient_record = json.dumps(json_data['Patient'], indent=2)
    # Remove sensitive information: "Admission_info", "Demographics"
    patient_record.pop("Admission_info", None)
    patient_record.pop("Demographics", None)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical assistant. Based only on the patient's medical chart below, "
                    "answer the user's question as if you are the patient. "
                    "Keep the answer direct and personal, in first person (e.g., 'I have...', 'My doctor gave me...')."
                    "If the information is not present in the chart, say 'I don't know' or 'I don't remember'."
                )
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nPatient Record:\n{patient_record}\n\nAnswer:"
            }
        ],
        temperature=0
    )
    return response.choices[0].message.content

def find_answer(question, json_data, relevant_data, model):
    flag = 1
    answer = generate_answer(question, relevant_data, model)
    if answer:
        flag = 0
        return answer, flag
    # Fallback if generate_answer returned None
    return fallback_llm_answer(question, json_data, model), flag

def generate_initial_patient_intro(json_data, model):
    patient = json_data["Patient"]
    demographics = patient.get("Demographics", {})
    admission_info = patient.get("Admission_info", {})
    chief_complaint = patient.get("Chief Complaint", "")

    structured_context = {
        "Demographics": demographics,
        "Admission Info": {
            "Admission Location": admission_info.get("admission_location", ""),
            "Preliminary Diagnosis": admission_info.get("admission_diagnosis", ""),
            "Admission Type": admission_info.get("admission_type", ""),
        },
        "Chief Complaint": chief_complaint,
    }

    system_msg = (
        "You are a patient introducing yourself to a doctor at the start of a hospital visit. "
        "Using the structured medical data provided, generate a short, realistic, and natural-sounding description "
        "of your condition in the first person (e.g., 'Hi, I'm a 60-year-old male...'). "
        "Mention relevant symptoms, history, and why you're at the hospital. Be concise and medically accurate."
    )

    user_prompt = (
        f"Patient Data:\n{json.dumps(structured_context, indent=2)}\n\n"
        "Begin your introduction:"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

def main(json_path, question=None, model="gpt-4o"):
    # Load JSON
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        true_diagnoses = json_data["Patient"].pop("Diagnoses", None)

    # Init mode: no question provided
    if not question:
        intro = generate_initial_patient_intro(json_data, model=model)
        # print(f"\nInitial Patient Introduction:\n{intro}")
        return intro

    # QA Pipeline
    keywords = extract_keywords(question)
    sections = match_keywords_to_sections(question, keyword_mapping)
    relevant_data = extract_data(json_data, sections, question)
    answer, flag = find_answer(question, json_data, relevant_data, model)

    # Output
    # print(f"Question: {question}")
    # print(f"Matched Sections: {sections}")
    # print(f"Relevant Data Extracted:\n{relevant_data}")
    # print(f"\nAnswer:\n{answer}")

    return answer, flag#, true_diagnoses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run patient QA agent.")
    parser.add_argument("--model", type=str, default="gpt-4.1", help="Model to use for LLM inference.")
    parser.add_argument("--path", type=str, required=True, help="Path to the patient JSON file.")
    parser.add_argument("--question", type=str, default=None, help="Question to ask the patient. If not provided, the agent introduces themselves.")

    args = parser.parse_args()
    main(args.path, args.question, model=args.model)


