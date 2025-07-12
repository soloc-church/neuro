from pathlib import Path
import pickle
from .hamiltonians import *
from .transpile_utils import run_qaoa   # placeholder, implement later

def run_or_load(tag: str, cost_op, **kwargs):
    cache = Path("data/derived") / f"{tag}.pkl"
    if cache.exists():
        return pickle.load(cache.open("rb"))
    result = run_qaoa(cost_op, **kwargs)
    pickle.dump(result, cache.open("wb"))
    return result
