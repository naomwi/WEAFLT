from PyEMD import EEMD
import numpy as np

def run_ceemd(data, trials=50, max_imfs=12):
    """
    Run CEEMD (Complete Ensemble EMD) decomposition.
    This is the base approach for Baselines model.
    """
    print(f"--> Running CEEMD (Trials: {trials})... Please wait.")
    eemd = EEMD(trials=trials, noise_width=0.2, parallel=False)

    imfs = eemd.eemd(data.reshape(-1), max_imf=max_imfs)

    return imfs

# Alias for backward compatibility
def run_ceemdan(data, trials=50, max_imfs=12):
    """Backward compatible alias - now uses CEEMD"""
    return run_ceemd(data, trials, max_imfs)