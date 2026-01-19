# execution_agent/repetition.py
import json
from typing import List, Dict, Any

def _canon(cmd: Dict[str, Any]) -> str:
    return json.dumps(cmd, sort_keys=True, ensure_ascii=False)

def is_repetition(last_cmds: List[Dict[str, Any]], candidate: Dict[str, Any]) -> bool:
    """
    Replicates your 6-window repetition checks:
      - A A A A A A
      - A B A B A B
      - A B C A B C
      - A A A B B B
    last_cmds: list of previous commands (dicts with name/args), most recent last.
    """
    window = (last_cmds[-5:] if len(last_cmds) >= 5 else last_cmds[:]) + [candidate]
    if len(window) < 6:
        return False

    s = [_canon(x) for x in window]  # length 6

    # period-1
    if len(set(s)) == 1:
        return True

    # period-2: A B A B A B
    if s[0] == s[2] == s[4] and s[1] == s[3] == s[5] and s[0] != s[1]:
        return True

    # period-3: A B C A B C
    if s[0] == s[3] and s[1] == s[4] and s[2] == s[5] and len({s[0], s[1], s[2]}) == 3:
        return True

    # AAA BBB
    if s[0] == s[1] == s[2] and s[3] == s[4] == s[5] and s[0] != s[3]:
        return True

    return False
