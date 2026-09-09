import json, sys
sys.stdout.reconfigure(encoding='utf-8')

# The diagnostics paragraph follows the 29-session session-disjoint split description
# -> check the label_quality.json of the matching training run
d = json.load(open(r'D:\reloc3r\out_train_fixed_seq_20260305_015535\label_quality.json'))
o = d.get('overall', d)
def g(k):
    v = o.get(k, {})
    return f"mean={v.get('mean', 0):.4g} std={v.get('std', 0):.4g} p95={v.get('p95', 0):.4g} max={v.get('max', 0):.4g} p90={v.get('p90', 0):.4g}"
print('imu_end_dt:      ', g('imu_end_dt'))
print('imu_max_gap:     ', g('imu_max_gap'))
print('pose_start_gap:  ', g('pose_start_gap_s'))
print('reproj_rmse_end: ', g('reproj_rmse_end'))
print()
print('Manuscript:')
print('  imu_end_dt mean/std 0.00 +- 3.47e-3')
print('  imu_max_gap p95/max 0.01563/0.01628')
print('  pose_start_gap mean/std 0.171+-0.00768, p95/max 0.1775/0.1914')
print('  reproj end mean/p90/p95 = 1.258/1.781/1.866 px')
