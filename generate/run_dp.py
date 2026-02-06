import pandas as pd
from dp_selection import run_dp_selection

df = pd.read_csv("/media/datassd/sina/libgen/generate/data/sweep_results.csv")
paths = run_dp_selection(
    df,
    out_prefix="/media/datassd/sina/libgen/generate/outputs/dp",   # change if you want
    prune=False,                     # pruning infra exists; keep False for this first run
    V_values=[2,3,4,5]
)
print(paths)