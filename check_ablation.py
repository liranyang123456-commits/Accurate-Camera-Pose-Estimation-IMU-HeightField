import pandas as pd, sys, json
sys.stdout.reconfigure(encoding='utf-8')

base = r'D:\reloc3r\Data_IMU_Camera_Pose_5'
for v in ['Ablation_Full', 'Ablation_wo_posegraph', 'Ablation_wo_quality_gate']:
    df = pd.read_csv(rf'{base}\{v}\diag.csv')
    fit = pd.to_numeric(df['pair_fitness'], errors='coerce').dropna()
    rmse = pd.to_numeric(df['pair_rmse'], errors='coerce').dropna()
    modes = df['gate_mode'].value_counts().to_dict() if 'gate_mode' in df.columns else {}
    print(f"{v:28s} frames={len(df)}  fitness_mean={fit.mean():.3f}  rmse_mean={rmse.mean():.3f}  gates={modes}")
