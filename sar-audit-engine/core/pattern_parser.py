import json
import os
from datetime import datetime

def parse_patterns_file(file_path, output_folder="data/processed/cases"):
    cases = []
    current_case = None
    case_counter = 1

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            # Detect beginning of laundering block
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                typology = line.split("-")[-1].strip()
                typology = typology.split(":")[0].strip()

                current_case = {
                    "case_id": f"CASE_{case_counter:03}",
                    "typology": typology,
                    "transactions": []
                }

            # Detect end of laundering block
            elif line.startswith("END LAUNDERING ATTEMPT"):
                if current_case:
                    cases.append(current_case)
                    case_counter += 1
                    current_case = None

            # Transaction lines
            elif current_case and line:
                parts = line.split(",")

                if len(parts) >= 10:
                    transaction = {
                        "timestamp": parts[0],
                        "from_bank": parts[1],
                        "from_account": parts[2],
                        "to_bank": parts[3],
                        "to_account": parts[4],
                        "amount_received": float(parts[5]),
                        "receiving_currency": parts[6],
                        "amount_paid": float(parts[7]),
                        "payment_currency": parts[8],
                        "payment_format": parts[9]
                    }

                    current_case["transactions"].append(transaction)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save each case as separate JSON
    for case in cases:
        file_name = f"{case['case_id']}.json"
        with open(os.path.join(output_folder, file_name), "w") as outfile:
            json.dump(case, outfile, indent=4)

    print(f"Parsed {len(cases)} laundering cases successfully.")
    return cases
