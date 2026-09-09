import pandas as pd, sys, os
sys.stdout.reconfigure(encoding='utf-8')

# manuscript: test1(n=900), test2(n=972), test3(n=790), test4(n=430), test5(n=502)
for i in [1, 2, 3, 4, 5]:
    p = rf'D:\reloc3r\Data_IMU_Camera_Pose_{i}\Ours_Camera_Pose_60f_eval_vis\diag.csv'
    if os.path.exists(p):
        n = len(pd.read_csv(p))
        print(f'Data_IMU_Camera_Pose_{i}: diag.csv rows = {n}')
    else:
        print(f'Data_IMU_Camera_Pose_{i}: diag.csv NOT FOUND')
# line2
p = r'D:\reloc3r\Data_IMU_Camera_Pose_5\Line2_Ours_Camera_Pose_60f_eval_vis\diag.csv'
if os.path.exists(p):
    print('Line2 diag.csv rows =', len(pd.read_csv(p)))
