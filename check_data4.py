import json, sys
sys.stdout.reconfigure(encoding='utf-8')

d = json.load(open(r'D:\reloc3r\out_baselines_fixedtest_from_out_train_20260310_031242.json'))
m = d['metrics']
print('methods in metrics:', list(m.keys()) if isinstance(m, dict) else type(m))
print()
if isinstance(m, dict):
    for k, v in m.items():
        if isinstance(v, dict):
            rot = v.get('rot_deg', {})
            tr = v.get('trans', {})
            hit = v.get('hit_rate', {})
            if rot and isinstance(rot, dict):
                print(f"{k}: N={v.get('N')}, rot mean={rot.get('mean'):.3f} p50={rot.get('p50'):.3f} p95={rot.get('p95'):.3f}, "
                      f"trans mean={tr.get('mean'):.3f} p50={tr.get('p50'):.3f} p95={tr.get('p95'):.3f}, "
                      f"hit_rot5={hit.get('rot_deg<=5', 0):.3f} hit_trans20={hit.get('trans<=20', 0):.3f}")
            else:
                print(k, '->', json.dumps(v)[:200])
