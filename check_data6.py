import json, sys
sys.stdout.reconfigure(encoding='utf-8')

d = json.load(open(r'D:\reloc3r\out_train_fixed_seq_20260310_031242\test_metrics.json'))
print('=== GRU6D test_metrics.json ===')
print(json.dumps(d, indent=1)[:2500])
