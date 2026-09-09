import json, sys
sys.stdout.reconfigure(encoding='utf-8')

for fn in ['out_baselines_fixedtest_chessboard_all_full_20260310_031242.json',
           'out_baselines_fixedtest_chessboard_all_20260310_031242.json']:
    print('=' * 20, fn, '=' * 20)
    d = json.load(open(r'D:\reloc3r\\' + fn))
    def show(m, prefix=''):
        if isinstance(m, dict):
            rot = m.get('rot_deg', {})
            tr = m.get('trans', {})
            hit = m.get('hit_rate', {})
            if isinstance(rot, dict) and 'mean' in rot:
                print(f"{prefix}: N={m.get('N')}, rot {rot.get('mean'):.3f}/{rot.get('p50'):.3f}/{rot.get('p95'):.3f}, "
                      f"trans {tr.get('mean'):.3f}/{tr.get('p50'):.3f}/{tr.get('p95'):.3f}, "
                      f"hit {hit.get('rot_deg<=5', 0):.3f}/{hit.get('trans<=20', 0):.3f}")
    if isinstance(d, dict):
        for k, v in d.items():
            if k in ('config',):
                continue
            if isinstance(v, dict) and 'rot_deg' in v:
                show(v, k)
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    show(v2, f'{k}/{k2}')
    print()
