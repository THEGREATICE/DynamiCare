import pandas as pd
import json
import re
import time
import copy
import os
from openai import OpenAI
from answer_agent import main

with open("config.json") as f:
    config = json.load(f)
client = OpenAI(api_key=config["OPENAI_API_KEY"])

def log_event(log_path, role, func_name, input_text, output_text, extra=None):
    event = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "role": role,
        "function": func_name,
        "input": input_text,
        "output": output_text,
        "metadata": extra or {}
    }

    # Write to disk immediately
    with open(log_path, "a") as f:
        f.write(json.dumps(event) + "\n")

def json_safe_loads(content, required_keys=None):
    try:
        content = content.strip()
        # Extract from first { to last }
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != -1:
            content = content[start:end]
        parsed = json.loads(content)
        if required_keys:
            missing = [key for key in required_keys if key not in parsed]
            if missing:
                raise ValueError(f"Missing required keys: {missing}")
        return parsed
    
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decoding error: {e}")
    
def central(log_path, current_info, spec_list, model):
    response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": (
                """
                You are a central medical coordinator overseeing a diagnostic team. 
                Based on the current patient case and the specialists already involved, decide if additional experts are needed or if any can be removed.
                Only suggest adding new specialists if there's missing domain knowledge. Only suggest removing specialists if their role has been fully covered or no longer needed.
                The maximum number of specialists is 5.
                Respond in this JSON format only:
                {
                "ADD": [<specialists to add>],
                "REMOVE": [<specialists to remove>],
                "UPDATED_LIST": [<updated specialists>],
                "RATIONALE": "<short justification>"
                }
                """
                )
            },
            {
                "role": "user",
                "content": f"Current Conversation: {current_info}\n\nCurrent Specialists:{spec_list}\n\nAnswer:"
            }
        ],
        temperature=0.3
    )
    central_decision = json_safe_loads(response.choices[0].message.content, required_keys=["ADD", "REMOVE", "UPDATED_LIST", "RATIONALE"])
    log_event(log_path, "central agent", "central", f"Current Conversation: {current_info}\n\nCurrent Specialists:{spec_list}", central_decision)

    return central_decision

def confidence_check(log_path, current_info, spec, model):
    response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": (
                f"""
                You are a {spec}. Based on the current patient case, decide whether you are confident enough to make a diagnosis or whether more information is needed.

                Choose between the following ratings: 
                "Very Confident"- The diagnosis is strongly supported by current information, and no major uncertainties remain.
                "Somewhat Confident"- The diagnosis is likely given the evidence, but a bit more information would increase certainty.
                "Neither Confident or Unconfident"- Some clues suggest a possible diagnosis, but key details are still missing. 
                "Somewhat Unconfident"- Several diagnoses remain plausible; more data is needed to narrow them down.
                "Very Unconfident"- There is too little evidence to form a reasonable diagnostic opinion.

                Respond in the following format only: 
                DECISION: chosen rating from the above list.
                """
                )
            },
            {
                "role": "user",
                "content": f"Current Conversation: {current_info}\n\nAnswer:"
            }
        ],
        temperature=0.3
    )
    confidence_decision = response.choices[0].message.content
    log_event(log_path, "single specialist", "confidence_check", f"Current Conversation: {current_info}", confidence_decision, {"specialist": spec})
    return confidence_decision

def solo_decision(log_path, current_info, spec, model):
    status = 0
    # confidence check
    confidence_decision = confidence_check(log_path, current_info, spec, model)
    confidence = confidence_decision.lower()
    if "very confident" in confidence or "somewhat confident" in confidence:
        response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"""
                    You are a {spec}. Based on the current patient case, list the top 10 most likely diagnoses for this patient.
                    Only include the **diagnosis name** in the list.
                    Respond in this JSON format only:
                    {{
                    "RESPONSE_TYPE": "diagnosis",
                    "RESPONSE_CONTENT": "[<your diagnosis list>]",
                    "RATIONALE": "<brief justification>"
                    }}
                    """
                    )
                },
                {
                    "role": "user",
                    "content": f"Current Conversation: {current_info}\n\nAnswer:"
                }
            ],
            temperature=0.3
        )
        solo_decision = json_safe_loads(response.choices[0].message.content, required_keys=["RESPONSE_TYPE", "RESPONSE_CONTENT", "RATIONALE"])
        log_event(log_path, "solo diagonsis", "solo_decision", f"Current Conversation: {current_info}", solo_decision, {"specialist": spec})
        status = 1
    else:
        response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"""
                    You are a {spec}. Based on your medical expertise and the current information, propose the most important next question that would help you narrow down or confirm a diagnosis.
                    The question should be specific and relevant to the case.
                    Do not repeat any questions from the previous conversation log or ask about information already provided.
                    Avoid asking about topics already answered with "I don't know" or "not in chart"
                    If referencing labs, vitals, ECG, radiology, etc. — which may have **multiple time points** — be clear about the **desired time window**.
                    Respond in this JSON format only:
                    {{
                    "RESPONSE_TYPE": "question",
                    "RESPONSE_CONTENT": "<your follow-up question>",
                    "RATIONALE": "<brief justification>"
                    }}
                    """
                    )
                },
                {
                    "role": "user",
                    "content": f"Current Conversation: {current_info}\n\nAnswer:"
                }
            ],
            temperature=0.3
        )
        solo_decision = json_safe_loads(response.choices[0].message.content, required_keys=["RESPONSE_TYPE", "RESPONSE_CONTENT", "RATIONALE"])
        log_event(log_path, "solo question", "solo_decision", f"Current Conversation: {current_info}", solo_decision, {"specialist": spec})
        
    return solo_decision, status

def collab_decision(log_path, current_info, spec_list, model, agree_threshold=0.5):
    agent_outputs = []
    status = 0

    # === Step 1: Each agent submits a proposal ===
    for spec in spec_list:
        response = client.chat.completions.create(
            model=model,
            messages=[
            {
                "role": "system",
                "content": (
                    f"""
                    You are a {spec}. Based on your medical expertise and the current information, respond with either:
                    - A list of the top 10 most likely diagnoses (if you believe you can make a clinical judgment now), or
                    - A follow-up question (if you believe more information is needed to proceed)
                    
                    Only include the **diagnosis name** in the list.
                    The question should be specific and relevant to the case.
                    Do not repeat any questions from the previous conversation log or ask about information already provided.
                    Avoid asking about topics already answered with "I don't know" or "not in chart"
                    If referencing labs, vitals, ECG, radiology, etc. — which may have **multiple time points** — be clear about the **desired time window**.

                    Respond in this JSON format only:
                    {{
                    "RESPONSE_TYPE": "<diagnosis | question>",
                    "RESPONSE_CONTENT": "<your diagnosis list or follow-up question>",
                    "CONFIDENCE": "<1-5>",  // 5 = Very confident on the appropriateness of the response,
                    "RATIONALE": "<brief justification>"
                    }}
                    """
                    )
                },
                {
                    "role": "user",
                    "content": f"Current Conversation: {current_info}\n\nAnswer:"
                }
            ],
            temperature=0.3
        )
        result = json_safe_loads(response.choices[0].message.content, required_keys=["RESPONSE_TYPE", "RESPONSE_CONTENT", "CONFIDENCE", "RATIONALE"])
        log_event(log_path, "multi specialist", "collab_decision", f"Current Conversation: {current_info}", result, {"specialist": spec})
        result["SPECIALIST"] = spec
        result["CONFIDENCE"] = float(result["CONFIDENCE"])
        agent_outputs.append(result)

    # === Step 2: Voting ===
    for candidate in sorted(agent_outputs, key=lambda x: -x["CONFIDENCE"]):
        # print(f"\n🗳 Voting on proposal by {candidate['SPECIALIST']} ({candidate['RESPONSE_TYPE']}):")
        # print(f"- Content: {candidate['RESPONSE_CONTENT']}")
        # print(f"- Confidence: {candidate['CONFIDENCE']}")
        # print(f"- Rationale: {candidate['RATIONALE']}\n")

        log_event(log_path, "voting", "collab_decision", candidate, "", {"specialist": candidate["SPECIALIST"]})

        agree_votes = 0
        voters = [a for a in agent_outputs if a["SPECIALIST"] != candidate["SPECIALIST"]]

        for voter in voters:
            response = client.chat.completions.create(
                model=model,
                messages=[
                {
                    "role": "system",
                    "content": (
                        f"""
                        You are a {voter['SPECIALIST']}. Another specialist ({candidate['SPECIALIST']}) proposed the following:

                        They suggested to proceed with a **{candidate['RESPONSE_TYPE']}**, which is:
                        "{candidate['RESPONSE_CONTENT']}"

                        Their rationale is:
                        "{candidate['RATIONALE']}"

                        Based on the current patient case, decide if you agree with the proposed next step.
                        Respond with ONLY one of the following options:
                        - AGREE
                        - DISAGREE
                        """
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Current Conversation: {current_info}\n\nAnswer:" 
                    }
            ],
                temperature=0.3
            )
            vote_raw = response.choices[0].message.content.strip().lower()
            log_event(log_path, voter['SPECIALIST'], "vote", candidate, vote_raw, {"specialist": voter["SPECIALIST"], "candidate": candidate["SPECIALIST"]})
            if "agree" in vote_raw:
                agree_votes += 1

        agreement_ratio = (agree_votes + 1) / len(spec_list)
        log_event(log_path, "vote result", "collab_decision", candidate, "", {"agree_votes": agree_votes, "total_votes": len(spec_list)})
        print(f"✅ Agreement: {agree_votes + 1}/{len(spec_list)} ({agreement_ratio:.2%})")

        if agreement_ratio >= agree_threshold:
            print("🎯 Consensus achieved.")
            status = 1 if candidate["RESPONSE_TYPE"].lower() == "diagnosis" else 0
            log_event(log_path, "consensus", "collab_decision", candidate, "", {"status": status})
            return candidate, status  # return the agreed-upon proposal

    # === Step 3: Fallback ===
    print("⚠️ No consensus. Using top confidence proposal.")
    fallback = max(agent_outputs, key=lambda x: x["CONFIDENCE"])
    status = 1 if fallback["RESPONSE_TYPE"].lower() == "diagnosis" else 0
    log_event(log_path, "fallback", "collab_decision", fallback, "", {"status": status})
    return fallback, status

def run_qa_loop(json_path, log_path, model, max_rounds):
    current_info = []

    # initialize
    json_id = re.search(r"(\d+)", json_path).group(0)
    patient_init = main(json_path, question=None, model=model) # get initial patient info from answer_agent
    current_info.append({"PATIENT_INTRODUCTION": patient_init})
    log_event(log_path, "patient initialize", "run_qa_loop", json_id, patient_init)
    # get initial triage
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    """
                    You are a general practitioner triaging a new patient. Based on the patient's initial admission information, recommend one or more medical specialists to consult.
                    The maximum number of specialists is 5.
                    Return your answer in the following JSON format only:
                        {
                        "RATIONALE": "<short justification>",
                        "SUGGEST_SPECIALISTS": [<list of specialists>]
                        }
                    """
                )
            },
            {
                "role": "user",
                "content": f"Patient Introduction:\n{patient_init}\n\nAnswer:" 
            }
        ],
        temperature=0.3
    )
    triage = json_safe_loads(response.choices[0].message.content, required_keys=["RATIONALE", "SUGGEST_SPECIALISTS"])
    log_event(log_path, "triage", "run_qa_loop", copy.deepcopy(current_info), triage)
    list_spec = triage["SUGGEST_SPECIALISTS"]

    # main QA LOOP
    for round_num in range(max_rounds):
        print(f"\n🔄 Round {round_num+1}")

        if not isinstance(list_spec, list):
            list_spec = [list_spec] if isinstance(list_spec, str) else []
        # Step 1: Decide how to process based on team size
        if len(list_spec) == 1:
            output, status = solo_decision(log_path, current_info, list_spec[0], model)
            if status == 1:
                # If solo decision is confident, we can end the loop
                print("✅ Solo Decision Made")
                log_event(log_path, "Solo Decision Made", "run_qa_loop", copy.deepcopy(current_info), output, {"status": status})
                return output, status
        else:
            output, status = collab_decision(log_path, current_info, list_spec, model, agree_threshold=0.5)
            if status == 1:
                # If collaboration is confident, we can end the loop
                print("✅ Collaboration Decision Made")
                log_event(log_path, "Collaboration Decision Made", "run_qa_loop", copy.deepcopy(current_info), output, {"status": status})
                return output, status
        
        # Step 2: Central agent reviews output and decides next team
        question = output["RESPONSE_CONTENT"]
        current_info.append({"QUESTION": question})
        answer, flag = main(json_path, question, model=model)
        current_info.append({"ANSWER": answer})
        log_event(log_path, "answer", "run_qa_loop", copy.deepcopy(current_info), answer, {"LLM fallback":flag}) # flag == 1 fallback, 0 normal
        decision = central(log_path, current_info, list_spec, model)

        # Update the list of specialists based on central decision
        list_spec = decision["UPDATED_LIST"]
    
    # If we reach here, we have exhausted the max rounds
    print("⚠️ Max rounds reached. No consensus.")
    log_event(log_path, "Max rounds reached", "run_qa_loop", copy.deepcopy(current_info), decision, {"status": status})
    # Make a final decision anyway
    if len(list_spec) == 1:
        print("🧠 Forcing final solo decision...")
        response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"""
                    You are a {list_spec[0]}. Based on the current patient case, list the top 10 most likely diagnoses for this patient.
                    Only include the **diagnosis name** in the list.
                    Respond in this JSON format only:
                    {{
                    "RESPONSE_TYPE": "diagnosis",
                    "RESPONSE_CONTENT": "[<your diagnosis list>]",
                    "RATIONALE": "<brief justification>"
                    }}
                    """
                    )
                },
                {
                    "role": "user",
                    "content": f"Current Conversation: {current_info}\n\nAnswer:"
                }
            ],
            temperature=0.3
        )
        decision = json_safe_loads(response.choices[0].message.content, required_keys=["RESPONSE_TYPE", "RESPONSE_CONTENT", "RATIONALE"])
        log_event(log_path, "Final solo decision", "run_qa_loop", copy.deepcopy(current_info), copy.deepcopy(decision), {"forced": True, "specialist": list_spec[0]})
    else:
        print("🧠 Forcing final collaborative decision...")
        agent_outputs = []

        # === Step 1: Each agent submits a proposal ===
        for spec in list_spec:
            response = client.chat.completions.create(
                model=model,
                messages=[
                {
                    "role": "system",
                    "content": (
                        f"""
                        You are a {spec}. Based on the current patient case, list the top 10 most likely diagnoses for this patient.
                        Only include the **diagnosis name** in the list.
                        Respond in this JSON format only:
                        {{
                        "RESPONSE_TYPE": "diagnosis",
                        "RESPONSE_CONTENT": "[<your diagnosis list>]",
                        "CONFIDENCE": "<1-5>",  // 5 = Very confident on the appropriateness of the response,
                        "RATIONALE": "<brief justification>"
                        }}
                        """
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Current Conversation: {current_info}\n\nAnswer:"
                    }
                ],
                temperature=0.3
            )
            result = json_safe_loads(response.choices[0].message.content, required_keys=["RESPONSE_TYPE", "RESPONSE_CONTENT", "CONFIDENCE", "RATIONALE"])
            log_event(log_path, "multi specialist", "collab_decision", f"Current Conversation: {current_info}", result, {"specialist": spec})
            result["SPECIALIST"] = spec
            result["CONFIDENCE"] = float(result["CONFIDENCE"])
            agent_outputs.append(result)

        # === Step 2: Voting ===
        for candidate in sorted(agent_outputs, key=lambda x: -x["CONFIDENCE"]):
            log_event(log_path, "voting", "collab_decision", candidate, "", {"specialist": candidate["SPECIALIST"]})

            agree_votes = 0
            voters = [a for a in agent_outputs if a["SPECIALIST"] != candidate["SPECIALIST"]]

            for voter in voters:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                    {
                        "role": "system",
                        "content": (
                            f"""
                            You are a {voter['SPECIALIST']}. Another specialist ({candidate['SPECIALIST']}) proposed the following:

                            They suggested to proceed with a **{candidate['RESPONSE_TYPE']}**, which is:
                            "{candidate['RESPONSE_CONTENT']}"

                            Their rationale is:
                            "{candidate['RATIONALE']}"

                            Based on the current patient case, decide if you agree with the proposed next step.
                            Respond with ONLY one of the following options:
                            - AGREE
                            - DISAGREE
                            """
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Current Conversation: {current_info}\n\nAnswer:" 
                        }
                ],
                    temperature=0.3
                )
                vote_raw = response.choices[0].message.content.strip().lower()
                log_event(log_path, voter['SPECIALIST'], "vote", candidate, vote_raw, {"specialist": voter["SPECIALIST"], "candidate": candidate["SPECIALIST"]})
                if "agree" in vote_raw:
                    agree_votes += 1

            agreement_ratio = (agree_votes + 1) / len(list_spec)
            log_event(log_path, "vote result", "collab_decision", candidate, "", {"agree_votes": agree_votes, "total_votes": len(list_spec)})
            print(f"✅ Agreement: {agree_votes + 1}/{len(list_spec)} ({agreement_ratio:.2%})")

            if agreement_ratio >= 0.5:
                print("🎯 Consensus achieved.")
                log_event(log_path, "consensus", "collab_decision", candidate, "", {"status": status})
                return candidate, status  # return the agreed-upon proposal

        # === Step 3: Fallback ===
        print("⚠️ No consensus. Using top confidence proposal.")
        fallback = max(agent_outputs, key=lambda x: x["CONFIDENCE"])
        log_event(log_path, "Final collab decision", "run_qa_loop", copy.deepcopy(current_info), fallback, {"forced": True, "status": status})
        return fallback, status


def main_doctor(json_path, model, max_rounds):
    json_id = re.search(r"(\d+)", json_path).group(0)
    log_path = f"data/log/log_{json_id}_{model}.json"
    result_path = f"data/result/result_{json_id}_{model}.json"
    output, status = run_qa_loop(json_path, log_path, model=model, max_rounds=max_rounds)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        true_diagnoses = json_data["Patient"].pop("Diagnoses", None)
    # Save the final output
    with open(result_path, "w") as f:
        json.dump({"output": output, "status": status, "ground_truth":true_diagnoses}, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Doctor Agent.")
    parser.add_argument("--path", type=str, help="Path to the JSON file containing patient data.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use.")
    parser.add_argument("--max_rounds", type=int, default=10, help="Maximum number of rounds for the QA loop.")
    args = parser.parse_args()
    if os.path.isdir(args.path):
        for fname in os.listdir(args.path):
            if fname.endswith(".json"):
                fpath = os.path.join(args.path, fname)
                print(f"Processing {fpath}...")
                main_doctor(fpath, model=args.model, max_rounds=args.max_rounds)

    #main_doctor(args.path, model=args.model, max_rounds=args.max_rounds)