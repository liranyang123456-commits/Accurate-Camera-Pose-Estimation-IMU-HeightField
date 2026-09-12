# 弱 B 采集协议（棋盘格 → 高纹理非棋盘格）

目的：在**同一条同步 IMU+相机序列**上，先对着棋盘格（有 PnP），再把镜头转到桌面/海报等非棋盘格纹理，观察旧 GRU6D 预测的相对位姿是否连续、不发散。

这是定性实验，**不能**报无棋盘格 ATE。

现有 2026-09-11 序列不能当弱 B：`pnp_seq_20260911_011115` 仅 7 帧 `board_left=0`，两条 VIO 序列全程 `board_left=1`。

已采集：`D:\reloc3r\Data_IMU_Camera_Pose_Sequences\stereo_endoscope\vio_seq_20260912_104835`（839 帧，85.9 s；离线 PnP 99/839，末次 12x9 在 43.3 s；其后键盘段无 PnP）。评测：

```text
python E:\elsarticle-templateCMBP_IMU_Camera\PR_R1_Revision\experiments\eval_gru6d_weak_b.py --session_dir D:\reloc3r\Data_IMU_Camera_Pose_Sequences\stereo_endoscope\vio_seq_20260912_104835 --skip_pnp
```

## 设备

与 2026-09-11 相同：内镜 cam0 + IMU，尽量不要拆安装。

## 动作（约 60–90 秒）

1. 开始录制后立刻运动，不要静置。
2. **前 20–30 秒**：对着 GP050-3（12×9，3 mm）棋盘格，三轴小幅运动（与训练分布接近）。
3. **中间不停机**：镜头平移离开棋盘格，对准键盘、海报、书脊等**高纹理、非棋盘格**平面/近平面。
4. 再运动 20–30 秒。
5. 停止。

## 采集命令

沿用现有 session 采集脚本（不要在线 PnP）：

```text
python D:\reloc3r\IMU_Camera_Pose_MLP_Predict_Sequences.py --no_pose
```

把新 session 目录记下来，例如 `D:\reloc3r\Data_IMU_Camera_Pose_Sequences\stereo_endoscope\weakB_YYYYMMDD_HHMMSS`。

## 评测（采集完成后）

```text
python E:\elsarticle-templateCMBP_IMU_Camera\PR_R1_Revision\experiments\eval_gru6d_weak_b.py --session_dir <新session>
```

输出：

- PnP 覆盖率（棋盘格段）
- 链式相对位姿的平移/旋转范数曲线
- 棋盘格可见 vs 不可见两段的统计
- 图：`weakB_traj.png`
