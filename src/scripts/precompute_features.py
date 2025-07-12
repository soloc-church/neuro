from pathlib import Path
import numpy as np
from src.dataio.loaders import load_session
from src.features import pearson_corr_matrix, theta_gamma_mi_all

DERIVED = Path("data/derived")
DERIVED.mkdir(parents=True, exist_ok=True)

for sess in Path("data/raw").glob("session*.npy"):
    name = sess.stem
    traces = np.load(sess)
    corr = pearson_corr_matrix(traces)
    np.save(DERIVED / f"{name}_corr.npy", corr)

    theta_phase, gamma_pow = load_session(name, which="oscillations")
    mi_vals = theta_gamma_mi_all(theta_phase, gamma_pow)
    np.save(DERIVED / f"{name}_mi.npy", mi_vals)

    print(f"Completed {name}")
