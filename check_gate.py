import pandas as pd, sys, os
sys.stdout.reconfigure(encoding='utf-8')

candidates = [
    r'D:\reloc3r\Data_IMU_Camera_Pose_5\Ours_Camera_Pose_60f_eval_vis\diag.csv',
    r'D:\reloc3r\Data_IMU_Camera_Pose_5\Ablation_Full\diag.csv',
]
for path in candidates:
    if not os.path.exists(path):
        print('NOT FOUND:', path)
        continue
    df = pd.read_csv(path)
    print('===', path)
    print('rows:', len(df))
    if 'gate_mode' in df.columns:
        print('gate_mode counts:')
        print(df['gate_mode'].value_counts().to_string())
    if 'gate_jump' in df.columns:
        print('gate_jump counts:', df['gate_jump'].value_counts().to_dict())
    for col in ['fitness', 'rmse']:
        if col in df.columns:
            print(f'{col}: mean={pd.to_numeric(df[col], errors="coerce").mean():.3f}')
    print()
