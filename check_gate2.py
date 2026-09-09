import pandas as pd, sys, os
sys.stdout.reconfigure(encoding='utf-8')

path = r'D:\reloc3r\Data_IMU_Camera_Pose_5\Ours_Camera_Pose_60f_eval_vis\diag.csv'
df = pd.read_csv(path)
print('columns:', [c for c in df.columns if any(k in c.lower() for k in ['fit','rmse','gate','mode'])])
for col in df.columns:
    if 'fitness' in col.lower() or 'rmse' in col.lower():
        v = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(v):
            print(f'{col}: mean={v.mean():.3f} median={v.median():.3f} n={len(v)}')
print()

# geometric validity: dense_match_eval_on_chessboard.csv
for p in [r'D:\reloc3r\Data_IMU_Camera_Pose_5\Ours_Camera_Pose_60f_eval_vis\dense_match_eval_on_chessboard.csv',
          r'D:\reloc3r\Data_IMU_Camera_Pose_5\Ablation_Full\dense_match_eval_on_chessboard.csv']:
    if os.path.exists(p):
        d = pd.read_csv(p)
        print('===', os.path.dirname(p).split('\\')[-1], 'dense_match_eval')
        print('columns:', list(d.columns)[:15])
        print(d.describe().to_string())
        break
