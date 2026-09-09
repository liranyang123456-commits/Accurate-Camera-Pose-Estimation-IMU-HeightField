import json, sys, glob, os
sys.stdout.reconfigure(encoding='utf-8')

# Find the training run matching Table (imu_cam_results_one_table):
# Val: N=947, rot 3.10+-2.09, p50/p90 2.60/5.75, trans 9.42+-5.06 (8.57/15.88)
# Test: N=724, rot 2.24+-1.59, p50/p90 1.83/4.11, trans 10.22+-5.07 (9.51/18.02)
roots = glob.glob(r'D:\reloc3r\out_train_*\val_metrics.json')
print('scanning', len(roots), 'training runs...')
for vf in roots:
    try:
        v = json.load(open(vf))
        if v.get('N') == 947:
            r, t = v['rot_deg'], v['trans']
            print('MATCH N=947:', os.path.dirname(vf))
            print(f"  val rot {r['mean']:.2f}+-{r['std']:.2f} p50/p90 {r['p50']:.2f}/{r['p90']:.2f} | trans {t['mean']:.2f}+-{t['std']:.2f} ({t['p50']:.2f}/{t['p90']:.2f})")
            tf = os.path.join(os.path.dirname(vf), 'test_metrics.json')
            if os.path.exists(tf):
                te = json.load(open(tf))
                r2, t2 = te['rot_deg'], te['trans']
                print(f"  test N={te.get('N')} rot {r2['mean']:.2f}+-{r2['std']:.2f} p50/p90 {r2['p50']:.2f}/{r2['p90']:.2f} | trans {t2['mean']:.2f}+-{t2['std']:.2f} ({t2['p50']:.2f}/{t2['p90']:.2f})")
    except Exception as e:
        pass
