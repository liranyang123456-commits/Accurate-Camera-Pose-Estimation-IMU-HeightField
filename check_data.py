import json, sys
sys.stdout.reconfigure(encoding='utf-8')

vals = [2.4928, 3.2251, 3.6035, 18.7538, 0.4985]
print('opt_rot mean = %.4f  (manuscript Table 7 says 5.81)' % (sum(vals)/5))
raw = [2.0663, 2.3518, 3.3121, 3.7836, 0.5056]
print('raw_rot mean = %.4f  (manuscript Table 7 says 2.40)' % (sum(raw)/5))
raw_ate = [60.84, 95.57, 83.04, 105.44, 175.90]
opt_ate = [51.66, 90.84, 81.99, 104.46, 162.95]
print('raw_ate mean = %.2f  (ms: 104.16)' % (sum(raw_ate)/5))
print('opt_ate mean = %.2f  (ms: 98.38)' % (sum(opt_ate)/5))
print('drift reduction = %.2f%%  (ms: 5.5%%)' % ((sum(raw_ate)-sum(opt_ate))/sum(raw_ate)*100))
print()

d = json.load(open('data/results/vio_orbslam3_mono_fixedtest_metrics.json'))
print('=== vio_orbslam3_mono_fixedtest_metrics.json ===')
print(json.dumps(d, indent=1)[:3000])
