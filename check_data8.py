import json, sys
sys.stdout.reconfigure(encoding='utf-8')

# 1. Per-session ORB-SLAM3 (Table 4)
d = json.load(open(r'D:\reloc3r\vio_orbslam3_mono_fixedtest_metrics.json'))
print('=== Per-session ORB-SLAM3 (Table 4 check) ===')
for sess, v in d['per_session'].items():
    n = v.get('N', v.get('rpe_delta_window', {}).get('N', '?'))
    rm = v.get('rot_deg_mean', v.get('rpe_delta_window', {}).get('rot_deg_mean', '?'))
    traj = 'yes' if v.get('ate', {}).get('N') else '?'
    print(f"{sess}: RPE_N={n}, rot_mean={rm if rm=='?' else round(rm,3)}, traj={traj}")
print()

# 2. Session-disjoint split metrics (Table imu_cam_results_one_table)
v = json.load(open(r'D:\reloc3r\out_train_fixed_seq_20260310_031242\val_metrics.json'))
print('=== val_metrics.json (Val split) ===')
print('N =', v.get('N'))
r, t = v.get('rot_deg', {}), v.get('trans', {})
print(f"rot mean/std = {r.get('mean'):.3f}/{r.get('std'):.3f}, p50/p90 = {r.get('p50'):.3f}/{r.get('p90'):.3f}")
print(f"trans mean/std = {t.get('mean'):.3f}/{t.get('std'):.3f}, p50/p90 = {t.get('p50'):.3f}/{t.get('p90'):.3f}")
