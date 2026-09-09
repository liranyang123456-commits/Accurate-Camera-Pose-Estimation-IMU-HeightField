import re, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'D:\reloc3r\Data_IMU_Camera_Pose_5\Ours_Camera_Pose_60f_eval_vis\console_output.log'
raw = open(path, 'rb').read().decode('utf-8', errors='replace')

# separate categories: lines contain category before 总耗时
cats = {}
for m in re.finditer(r'\[时间统计\]\s*([^:：]+)[:：][^\n]*?总耗时\s*([\d.]+)', raw):
    cat = re.sub(r'\([^)]*\)', '', m.group(1)).strip()
    cat = re.sub(r'\d+', '', cat)
    cats.setdefault(cat, []).append(float(m.group(2)))

for cat, vals in cats.items():
    vals_s = sorted(vals)
    n = len(vals_s)
    mean = sum(vals_s)/n*1000
    med = (vals_s[n//2] if n%2 else (vals_s[n//2-1]+vals_s[n//2])/2)*1000
    p95 = vals_s[min(n-1, int(n*0.95))]*1000
    print(f'{cat[:30]:32s} n={n:5d}  mean={mean:7.0f}ms  median={med:7.0f}ms  p95={p95:7.0f}ms')
print()
print('Manuscript: height-field construction mean/median/p95 = 2186/2183/2445 ms')
