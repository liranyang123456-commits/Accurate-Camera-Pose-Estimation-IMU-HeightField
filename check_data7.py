import json, sys
sys.stdout.reconfigure(encoding='utf-8')

for name in ['orb_vo_fixedtest_metrics.json', 'orb_imu_quat_fixedtest_metrics.json']:
    d = json.load(open(r'D:\reloc3r\vio_runs_classical\\' + name))
    print('=' * 15, name, '=' * 15)
    m = d.get('metrics', d)
    rot = m.get('rot_deg', {})
    tr = m.get('trans', {})
    hit = m.get('hit_rate', {})
    print(f"N={m.get('N')}")
    print(f"rot   mean/p50/p95 = {rot.get('mean'):.3f} / {rot.get('p50'):.3f} / {rot.get('p95'):.3f}")
    print(f"trans mean/p50/p95 = {tr.get('mean'):.3f} / {tr.get('p50'):.3f} / {tr.get('p95'):.3f}")
    print(f"hit rot5={hit.get('rot_deg<=5', 0):.3f}  trans20={hit.get('trans<=20', 0):.3f}")
    print()
