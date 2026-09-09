import re, sys
sys.stdout.reconfigure(encoding='utf-8')

# Manuscript: height-field construction mean/median/p95 = 2186/2183/2445 ms per frame on 960x540
# Log lines like: [时间统计] 图像处理 (第N帧): 总耗时 X.XX秒 — encoding may be garbled; match numbers
path = r'D:\reloc3r\Data_IMU_Camera_Pose_5\Ours_Camera_Pose_60f_eval_vis\console_output.log'
raw = open(path, 'rb').read().decode('utf-8', errors='replace')
# find all "总耗时 X.XX" patterns (seconds)
vals = [float(m.group(1)) for m in re.finditer(r'总耗时\s*([\d.]+)', raw)]
print('n timing records:', len(vals))
if vals:
    import statistics
    vals_ms = [v*1000 for v in vals]
    vals_ms.sort()
    n = len(vals_ms)
    print('mean = %.0f ms' % (sum(vals_ms)/n))
    print('median = %.0f ms' % (vals_ms[n//2] if n%2 else (vals_ms[n//2-1]+vals_ms[n//2])/2))
    print('p95 = %.0f ms' % vals_ms[int(n*0.95)])
print('Manuscript: mean/median/p95 = 2186/2183/2445 ms')
