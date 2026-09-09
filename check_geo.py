import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

p = r'D:\reloc3r\Data_IMU_Camera_Pose_5\Line2_Ours_Camera_Pose_60f_eval_vis\dense_match_eval_on_chessboard.csv'
d = pd.read_csv(p)
print('rows:', len(d))
print('err_mean_px:  mean=%.3f median=%.3f p95=%.3f' % (d['err_mean_px'].mean(), d['err_mean_px'].median(), d['err_mean_px'].quantile(0.95)))
print('err_median_px: mean=%.3f' % d['err_median_px'].mean())
print('ok_ratio_tau: mean=%.4f' % d['ok_ratio_tau'].mean())
print('tau_px values:', d['tau_px'].unique())
print()
print('Manuscript claims: mean/median/p95 of {e_i} = 0.511/0.385/0.806 px, rho_ok = 0.997, tau=3px, 99 pairs')
