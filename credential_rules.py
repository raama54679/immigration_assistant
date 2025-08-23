# credential_rules.py
from typing import List, Tuple


# Very simple, editable mapping. You can expand this.
# Keys are normalized desired outcomes; values list the *typical* docs you might want in hand for triage.
# ⚠️ Informational only – not legal advice.
REQUIREMENTS = {
    "work authorization": ["passport", "i-94", "job offer"],
    "student visa": ["passport", "sevis i-20", "financial support"],
    "lawful permanent residence": ["passport", "sponsor", "birth certificate"],
    "citizenship": ["green card", "tax returns", "selective service (if applicable)"]
}


ALIASES = {
    "green card": "lawful permanent residence",
    "lpr": "lawful permanent residence",
    "citizen": "citizenship",
    "opt": "work authorization",
}


def normalize_desired_outcome(x: str) -> str:
    key = x.strip().lower()
    return ALIASES.get(key, key)




def check_sufficiency(desired_outcome: str, provided: List[str]) -> Tuple[bool, List[str]]:
    provided_set = {p.lower().strip() for p in provided}
    req = REQUIREMENTS.get(desired_outcome, [])
    missing = [r for r in req if r not in provided_set]
    return (len(missing) == 0, missing)