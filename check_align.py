import json, sys, glob, os
sys.stdout.reconfigure(encoding='utf-8')

# Manuscript alignment diagnostics:
# end-of-window IMU offset mean/std 0.00 +- 3.47e-3 s
# within-window max gap p95/max 0.01563/0.01628 s
# start-pose interpolation gap mean/std 0.171 +- 0.00768 s (p95/max 0.1775/0.1914 s)
# Chessboard reprojection RMSE at window end: mean/p90/p95 = 1.258/1.781/1.866 px

for f in glob.glob(r'D:\reloc3r\out_train_fixed_seq_20260310_031242\*.json'):
    d = json.load(open(f))
    s = json.dumps(d)
    if '3.47' in s or '0.01563' in s or '1.258' in s or '0.171' in s:
        print('=== MATCH in', os.path.basename(f))
        print(s[:2000])
        print()
