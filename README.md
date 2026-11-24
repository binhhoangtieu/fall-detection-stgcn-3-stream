# Three-Stream ST-GCN for Skeleton-Based Fall Detection under Subject-Independent Evaluation

Code and processed data for the paper

> **Three-Stream Spatio-Temporal Graph Convolutional Networks for Skeleton-Based Fall Detection under Subject-Independent Evaluation**
> Son Nguyen Van, Toan Nguyen Viet, Phong Bui Hai, Binh Hoang Tieu (manuscript under review).

The repository contains the pose-extraction script, the training and evaluation notebooks, the processed skeleton sequences of the MV-Fall dataset with participant and camera-view labels, and the scripts that produce the figures and latency measurements of the paper.

Raw videos are **not** part of this repository, to protect the privacy of the participants. They are available from the corresponding author on reasonable request.

## Main results

MV-Fall, video level, mean ± standard deviation over five seeds.


## Repository layout

```
extract_keypoint_v2.py        Pose extraction (MediaPipe Pose) -> skeleton windows + metadata
keypoints_v2/
  custom_all_v2.npz           Processed MV-Fall skeleton sequences (see "Data format")
  videos_v2_public.csv        Per-clip metadata (anonymised)
colab/
  02_train_protocols_v2.ipynb Full model under the three protocols (E0 random split, E1 LOSO, E3 cross-view)
  03_baselines_ablation_E1.ipynb  Baselines and ablation, part 1 (ST-GCN, 2s-AGCN, LSTM, J+M, no skip, no adaptive, raw bone)
  04_ablation_tierB_E1.ipynb  Ablation, part 2 (J, M, A, J+A, M+A, J+M without adaptive, GRU)
  01_extract_keypoints_v2.ipynb   Pose extraction on Colab (optional alternative to the script)
tools/
  make_qualitative_figure.py  Skeleton-sequence figure of correct / missed / false-alarm windows
  measure_latency.py          CPU latency of pose estimation and of the classifier (ONNX Runtime)
  a3s_stgcn_full.onnx         Full model exported to ONNX (one leave-one-subject-out fold)
figures_revision/             Figures generated for the paper
results_v2/                   Per-run metrics and per-clip predictions (pixel/, unit/, ablation/)
```

`extract_keypoint.py`, `train.ipynb`, `custom_all.npz` and `best_model.pth` belong to the earlier random-split pipeline and are kept only for reference; they are not used for the results above.

## Data format

`keypoints_v2/custom_all_v2.npz` holds 760 windows of 60 frames (2 s at 30 FPS) from 655 clips of three participants.

| Key | Shape | Content |
|---|---|---|
| `X_joint` | (760, 60, 14, 3) | pixel coordinates `x`, `y` and pose confidence `c` of 14 joints |
| `X_motion`, `X_angle` | (760, 60, 14, 3) | pre-computed motion and bone-direction streams (recomputed online during training) |
| `y` | (760,) | 1 = fall, 0 = non-fall |
| `subject` | (760,) | anonymised participant ID: `S1`, `S2`, `S3` |
| `view` | (760,) | `cam1` = eye-level view, `cam2` = elevated oblique view |
| `category` | (760,) | `Lying_Falls`, `Sitting_Falls`, `Standing_Falls`, `non_fall` |
| `nonfall_type` | (760,) | `hard` (fall-like action), `adl`, or `unspecified` |
| `video_id`, `window_idx`, `group_id` | (760,) | clip index, window index within the clip, grouping key used for the validation split |
| `img_w`, `img_h`, `orientation` | (760,) | frame size and `landscape` / `portrait` |

Joint order: 0 nose, 1/2 left/right shoulder, 3/4 left/right elbow, 5/6 left/right wrist, 7/8 left/right hip, 9/10 left/right knee, 11/12 left/right ankle, 13 mid-hip (root).

The leave-one-subject-out folds are defined by the `subject` field: each participant is the test set once. The validation split inside the training participants is produced deterministically by `group_val_split` in the notebooks (seed 42).

## Reproducing the results

Training was run on Google Colab (PyTorch 2.14.1, CUDA 13.0, one Tesla T4).

1. Copy `keypoints_v2/custom_all_v2.npz` and the notebooks in `colab/` to a Google Drive folder and set `ALL_NPZ` / `BASE_RESULTS` in the configuration cell to that folder.
2. Run `02_train_protocols_v2.ipynb` (GPU). Set `DRY_RUN = True` first to print the folds and run the leakage checks without training. Each run writes `metrics.json`, `predictions_window.csv` and `predictions_video.csv`; finished runs are skipped when the notebook is restarted.
3. Run `03_baselines_ablation_E1.ipynb` and `04_ablation_tierB_E1.ipynb` for the baselines and the ablation. Their last cells produce the summary table with McNemar and Wilcoxon tests.
4. `COORD_MODE = 'unit'` in the configuration cell repeats the experiments with image-size-normalised coordinates.

Pose extraction from videos (only needed with the raw videos):

```
pip install mediapipe==0.10.21 opencv-python pandas tqdm
python extract_keypoint_v2.py extract      # cached per video, can be resumed
python extract_keypoint_v2.py build        # -> keypoints_v2/custom_all_v2.npz, videos_v2.csv
python extract_keypoint_v2.py report       # sanity tables and skeleton overlays
```

Latency measurement and the qualitative figure:

```
pip install onnxruntime
python tools/measure_latency.py            # needs raw videos for the pose-estimation timing
python tools/make_qualitative_figure.py
```

## Evaluation protocols

- **E1, leave-one-subject-out (main):** train on two participants, test on the third; validation data come only from the training participants; horizontal flipping is an online augmentation of training data only.
- **E3, cross-view and cross-subject:** train on one camera view of two participants, test on the other view of the third.
- **E0, random window-level split:** windows are split at random and mirrored copies of all windows are added to the training set. This protocol leaks and is included only to quantify the bias (about 10 accuracy points).

## Citation

The paper is under review. Until it is published, please cite this repository:

```
@misc{mvfall_threestream_stgcn,
  title  = {Three-Stream Spatio-Temporal Graph Convolutional Networks for Skeleton-Based Fall Detection under Subject-Independent Evaluation},
  author = {Nguyen Van, Son and Nguyen Viet, Toan and Bui Hai, Phong and Hoang Tieu, Binh},
  year   = {2026},
  howpublished = {\url{https://github.com/binhhoangtieu/fall-detection-stgcn-3-stream}}
}
```

## Licence and contact

A licence file will be added to this repository. Until then, please contact the corresponding author before reusing the code or the data.

Corresponding author: Binh Hoang Tieu, CMC University, Hanoi, Vietnam (htbinh@cmcu.edu.vn).
