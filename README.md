# Three-Stream ST-GCN for Skeleton-Based Fall Detection under Subject-Independent Evaluation

Code and processed data for the paper

> **Three-Stream Spatio-Temporal Graph Convolutional Networks for Skeleton-Based Fall Detection under Subject-Independent Evaluation**
> Son Nguyen Van, Toan Nguyen Viet, Phong Bui Hai, Binh Hoang Tieu (manuscript under review).

The repository contains the pose-extraction script, the training and evaluation notebooks, the processed skeleton sequences of the MV-Fall dataset with participant and camera-view labels, and the scripts that produce the figures and latency measurements of the paper.

## Repository layout

/data
  custom_all_v2.npz           Processed MV-Fall skeleton sequences (see "Data format")
  videos_v2_public.csv        Per-clip metadata (anonymised)
  
/code
  02_train_protocols_v2.ipynb Full model under the three protocols (E0 random split, E1 LOSO, E3 cross-view)
  03_baselines_ablation_E1.ipynb  Baselines and ablation, part 1 (ST-GCN, 2s-AGCN, LSTM, J+M, no skip, no adaptive, raw bone)
  04_ablation_tierB_E1.ipynb  Ablation, part 2 (J, M, A, J+A, M+A, J+M without adaptive, GRU)
  01_extract_keypoints_v2.ipynb   Pose extraction on Colab (optional alternative to the script)


## Data format

`custom_all_v2.npz` holds 760 windows of 60 frames (2 s at 30 FPS) from 655 clips of three participants.

Joint order: 0 nose, 1/2 left/right shoulder, 3/4 left/right elbow, 5/6 left/right wrist, 7/8 left/right hip, 9/10 left/right knee, 11/12 left/right ankle, 13 mid-hip (root).


## Evaluation protocols


## Licence and contact

A licence file will be added to this repository. Until then, please contact the corresponding author before reusing the code or the data.

Corresponding author: Binh Hoang Tieu, CMC University, Hanoi, Vietnam (htbinh@cmcu.edu.vn).
