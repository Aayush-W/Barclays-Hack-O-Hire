import pandas as pd

def enrich_case_with_transactions(case, transactions_index):

    enriched_transactions = []

    for tx in case["transactions"]:
        timestamp = pd.to_datetime(tx["timestamp"], errors="coerce")
        if pd.isna(timestamp):
            continue

        key = (
            timestamp,
            str(tx["from_bank"]),
            str(tx["from_account"]),
            str(tx["to_bank"]),
            str(tx["to_account"]),
            float(tx["amount_received"]),
        )

        try:
            match = transactions_index.loc[key]
        except KeyError:
            continue

        if isinstance(match, pd.DataFrame):
            enriched_transactions.append(match.iloc[0].to_dict())
        else:
            enriched_transactions.append(match.to_dict())

    case["enriched_transactions"] = enriched_transactions
    case["transaction_count"] = len(enriched_transactions)

    return case
