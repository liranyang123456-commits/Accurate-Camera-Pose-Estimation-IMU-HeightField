import json, sys
sys.stdout.reconfigure(encoding='utf-8')

d = json.load(open('data/results/vio_orbslam3_vs_classical_subset.json'))
print('=== vio_orbslam3_vs_classical_subset.json ===')
print(json.dumps(d, indent=1)[:2500])
