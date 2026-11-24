"""
extract_keypoint_v2.py
======================
Trích khung xương 3 luồng (joint / motion / angle) cho MV-Fall v2 (Dataset2), chạy trên máy cục bộ.
Chỉ lấy 3 người Giáp (S1), Minh (S2), Toàn (S3); bỏ Phong.

Logic trích xuất giữ nguyên như extract_keypoint.py cũ:
  MediaPipe Pose model_complexity=1, smooth_landmarks=False, CONF_THRESH=0.1,
  14 khớp (13 + Mid-Hip ảo), nội suy frame thiếu, cửa sổ 60 frame không chồng lấn, tọa độ pixel thô.
Thêm metadata để chia không rò rỉ: subject, view, video_id, group_id, window_idx,
nonfall_type, activity, orientation. Không tạo video lật (is_flip = 0).

Cách dùng (chạy trong thư mục fall-detection-stgcn-3-stream):
  python3 extract_keypoint_v2.py extract --limit 3        # thử 3 video
  python3 extract_keypoint_v2.py extract                  # trích toàn bộ (có cache, chạy lại sẽ tiếp tục)
  python3 extract_keypoint_v2.py build                    # cắt cửa sổ -> keypoints_v2/custom_all_v2.npz
  python3 extract_keypoint_v2.py report                   # in bảng kiểm tra + vẽ overlay

Kết quả: keypoints_v2/custom_all_v2.npz, keypoints_v2/videos_v2.csv, keypoints_v2/sanity/*.png
Chép custom_all_v2.npz (và videos_v2.csv) lên Google Drive: MyDrive/FallDetection_NEW/
"""
import argparse
import hashlib
import os
import pickle
import re
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE        = Path(__file__).resolve().parent
DATA_ROOT   = HERE / 'Dataset2'
OUT_DIR     = HERE / 'keypoints_v2'
CACHE_DIR   = OUT_DIR / 'kp_cache'
OUT_NPZ     = OUT_DIR / 'custom_all_v2.npz'
OUT_CSV     = OUT_DIR / 'videos_v2.csv'
SANITY_DIR  = OUT_DIR / 'sanity'
SUBJECT_MAP = {'giap': 'S1', 'minh': 'S2', 'toan': 'S3'}   # Phong không có -> bị bỏ qua

FPS                   = 30
SEQ_LEN               = 60
MIN_FRAMES            = 30
MIN_DETECTED_FRAMES   = 30
MIN_VALID_FRAME_RATIO = 0.5
FALL_STRIDE           = SEQ_LEN
CONF_THRESH           = 0.1
NUM_KEYPOINTS         = 14

mp_pose = mp.solutions.pose
KEYPOINT_ORDER = [
    'Nose', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip', 'Left Knee',
    'Right Knee', 'Left Ankle', 'Right Ankle',
    # 13: Mid-Hip (ảo) = (Left Hip + Right Hip) / 2
]
LANDMARK_MAP = {n: getattr(mp_pose.PoseLandmark, n.upper().replace(' ', '_')) for n in KEYPOINT_ORDER}
PARENT_MAP = {13: None, 7: 13, 8: 13, 1: 7, 2: 8, 0: 1, 3: 1, 4: 2,
              5: 3, 6: 4, 9: 7, 10: 8, 11: 9, 12: 10}
EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8),
         (7, 9), (9, 11), (8, 10), (10, 12), (7, 13), (8, 13)]


# ─────────────────────────── tiện ích ───────────────────────────
def norm_key(s: str) -> str:
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    return s.replace('đ', 'd').replace('Đ', 'D').lower().strip()


def infer_category(rel_parts):
    k = norm_key('/'.join(rel_parts[:-1]))
    if 'lying' in k:       return 'Lying_Falls', 1, ''
    if 'sitting' in k:     return 'Sitting_Falls', 1, ''
    if 'standing' in k:    return 'Standing_Falls', 1, ''
    if 'de nham' in k:     return 'non_fall', 0, 'hard'
    if 'thuong ngay' in k: return 'non_fall', 0, 'adl'
    if 'non' in k:         return 'non_fall', 0, 'unspecified'
    return None, None, None


# ─────────────────────── trích xuất (giữ nguyên) ───────────────────────
def extract_keypoints_single(landmarks, img_w, img_h):
    kps = []
    for name in KEYPOINT_ORDER:
        try:
            lm = landmarks.landmark[LANDMARK_MAP[name].value]
            kps.append((lm.x * img_w, lm.y * img_h,
                        getattr(lm, 'visibility', getattr(lm, 'presence', 0.0))))
        except Exception:
            return None
    arr = np.array(kps, dtype=np.float32)
    mid_xy = (arr[7, :2] + arr[8, :2]) / 2.0
    mid_conf = np.minimum(arr[7, 2], arr[8, 2])
    return np.vstack([arr, np.array([mid_xy[0], mid_xy[1], mid_conf], dtype=np.float32)])


def compute_motion(j):
    m = np.zeros_like(j); m[1:] = j[1:] - j[:-1]; m[:, :, 2] = j[:, :, 2]; return m


def compute_bone_angle(j):
    F, V, _ = j.shape
    angle = np.zeros((F, V, 3), dtype=np.float32)
    for v in range(V):
        p = PARENT_MAP[v]
        if p is None:
            angle[:, v, 0] = 1.0; angle[:, v, 2] = j[:, v, 2]
        else:
            th = np.arctan2(j[:, v, 1] - j[:, p, 1], j[:, v, 0] - j[:, p, 0])
            angle[:, v, 0] = np.cos(th); angle[:, v, 1] = np.sin(th)
            angle[:, v, 2] = np.minimum(j[:, v, 2], j[:, p, 2])
    return angle


def interpolate_missing_keypoint_frames(raw_kps):
    if not raw_kps:
        return None, None, 0
    F = len(raw_kps)
    arr = np.full((F, NUM_KEYPOINTS, 3), np.nan, dtype=np.float32)
    valid = np.zeros(F, dtype=bool)
    for i, kp in enumerate(raw_kps):
        if kp is not None:
            arr[i] = kp; valid[i] = True
    n_valid = int(valid.sum())
    if n_valid == 0:
        return None, None, 0
    idx = np.arange(F); vidx = idx[valid]
    filled = np.empty_like(arr)
    for v in range(NUM_KEYPOINTS):
        for c in range(3):
            filled[:, v, c] = np.interp(idx, vidx, arr[valid, v, c])
    return filled.astype(np.float32), valid, n_valid


def extract_raw_from_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return dict(skel=None, mask=None, img_w=0, img_h=0, n_frames=0, n_detected=0,
                    shape_mismatch=False, err='cannot_open')
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw, first_shape = [], None
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=False,
                      min_detection_confidence=CONF_THRESH, min_tracking_confidence=CONF_THRESH) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if first_shape is None:
                first_shape = frame.shape[:2]
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            raw.append(extract_keypoints_single(res.pose_landmarks, img_w, img_h)
                       if res.pose_landmarks else None)
    cap.release()
    shape_mismatch = first_shape is not None and (first_shape[1], first_shape[0]) != (img_w, img_h)
    out = dict(img_w=img_w, img_h=img_h, n_frames=len(raw),
               n_detected=int(sum(k is not None for k in raw)),
               shape_mismatch=bool(shape_mismatch), err='')
    if len(raw) < MIN_FRAMES:
        out.update(skel=None, mask=None, err='too_few_frames'); return out
    skel, mask, n_valid = interpolate_missing_keypoint_frames(raw)
    if skel is None or n_valid < MIN_DETECTED_FRAMES:
        out.update(skel=None, mask=None, err='too_few_detected'); return out
    out.update(skel=skel, mask=mask)
    return out


def _pad(arr):
    F = arr.shape[0]
    if F >= SEQ_LEN:
        return arr[:SEQ_LEN]
    return np.concatenate([arr, np.repeat(arr[-1:], SEQ_LEN - F, axis=0)])


def make_windows(skel, mask):
    F = len(skel); wins = []
    if F < MIN_FRAMES:
        return wins
    if F < SEQ_LEN:
        if _pad(mask).mean() >= MIN_VALID_FRAME_RATIO:
            wins.append((0, _pad(skel)))
        return wins
    p = 0
    while p + SEQ_LEN <= F:
        if mask[p:p + SEQ_LEN].mean() >= MIN_VALID_FRAME_RATIO:
            wins.append((p, skel[p:p + SEQ_LEN].copy()))
        p += FALL_STRIDE
    return wins


# ─────────────────────────── liệt kê video ───────────────────────────
def collect_videos():
    videos = []
    for subj_dir in sorted(DATA_ROOT.iterdir()):
        skey = norm_key(subj_dir.name)
        if not subj_dir.is_dir() or skey not in SUBJECT_MAP:
            continue
        for vf in sorted(subj_dir.rglob('*')):
            if vf.suffix.lower() not in ('.mp4', '.mov', '.avi'):
                continue
            cat, label, nf_type = infer_category(vf.relative_to(subj_dir).parts)
            if cat is None:
                print('  [WARN] không suy ra được nhãn:', vf); continue
            m = re.search(r'cam\s*(\d)', vf.stem, re.I)
            stem_nocam = re.sub(r'_?cam\s*\d', '', vf.stem, flags=re.I)
            act = re.match(r'^\d+_[A-Za-z]+_([A-Za-z]+)_', vf.stem)
            rel_path = unicodedata.normalize('NFC', vf.relative_to(DATA_ROOT).as_posix())
            videos.append(dict(
                path=str(vf), rel_path=rel_path, subject=SUBJECT_MAP[skey],
                view=f'cam{m.group(1)}' if m else 'unknown', category=cat, label=label,
                nonfall_type=nf_type, activity=act.group(1) if act else '',
                group_id=f'{SUBJECT_MAP[skey]}/{norm_key(stem_nocam)}',
                cache_key=hashlib.md5(rel_path.encode('utf-8')).hexdigest()))
    vdf = pd.DataFrame(videos)
    vdf['video_id'] = np.arange(len(vdf))
    assert (vdf.view != 'unknown').all(), 'Có video không có cam1/cam2 trong tên'
    return vdf


def cache_path(key):
    return CACHE_DIR / f'{key}.pkl'


def _work(args):
    path, cpath = args
    try:
        out = extract_raw_from_video(path)
    except Exception as e:
        out = dict(skel=None, mask=None, img_w=0, img_h=0, n_frames=0, n_detected=0,
                   shape_mismatch=False, err=f'exception: {e}')
    tmp = str(cpath) + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(out, f)
    os.replace(tmp, cpath)            # ghi nguyên tử: bị ngắt giữa chừng không để lại cache hỏng
    return path


# ─────────────────────────── các bước ───────────────────────────
def cmd_extract(args):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vdf = collect_videos()
    print(f'Video: {len(vdf)}  |  {vdf.groupby("subject").size().to_dict()}')
    todo = [(r.path, cache_path(r.cache_key)) for r in vdf.itertuples() if not cache_path(r.cache_key).exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f'Cần trích: {len(todo)}  |  workers={args.workers}', flush=True)
    t0 = time.time()
    if args.workers == 1:
        for a in tqdm(todo, mininterval=10):
            _work(a)
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            for _ in tqdm(as_completed([ex.submit(_work, a) for a in todo]), total=len(todo), mininterval=10):
                pass
    done = sum(cache_path(k).exists() for k in vdf.cache_key)
    print(f'Xong lượt này sau {(time.time() - t0) / 60:.1f} phút. Đã có cache: {done}/{len(vdf)}')


def cmd_build(args):
    vdf = collect_videos()
    missing = [r.rel_path for r in vdf.itertuples() if not cache_path(r.cache_key).exists()]
    if missing:
        sys.exit(f'Còn {len(missing)} video chưa trích, chạy lại bước extract. Ví dụ: {missing[:3]}')
    rows, J, M, A = [], [], [], []
    meta = {k: [] for k in ['y', 'img_w', 'img_h', 'seq_id', 'video_id', 'window_idx', 'subject', 'view',
                            'category', 'nonfall_type', 'activity', 'group_id', 'orientation', 'is_flip']}
    for r in tqdm(vdf.itertuples(), total=len(vdf), mininterval=10):
        out = pickle.load(open(cache_path(r.cache_key), 'rb'))
        wins = make_windows(out['skel'], out['mask']) if out['skel'] is not None else []
        orient = 'portrait' if out['img_h'] > out['img_w'] else 'landscape'
        rows.append(dict(video_id=r.video_id, rel_path=r.rel_path, subject=r.subject, view=r.view,
                         category=r.category, label=r.label, nonfall_type=r.nonfall_type,
                         activity=r.activity, group_id=r.group_id, img_w=out['img_w'], img_h=out['img_h'],
                         orientation=orient, n_frames=out['n_frames'], n_detected=out['n_detected'],
                         detected_ratio=out['n_detected'] / max(1, out['n_frames']),
                         shape_mismatch=out.get('shape_mismatch', False), n_windows=len(wins),
                         error=out['err'] or ('no_valid_window' if not wins else '')))
        for wi, (_, w) in enumerate(wins):
            J.append(w); M.append(compute_motion(w)); A.append(compute_bone_angle(w))
            for k, v in dict(y=r.label, img_w=out['img_w'], img_h=out['img_h'], seq_id=r.video_id,
                             video_id=r.video_id, window_idx=wi, subject=r.subject, view=r.view,
                             category=r.category, nonfall_type=r.nonfall_type, activity=r.activity,
                             group_id=r.group_id, orientation=orient, is_flip=0).items():
                meta[k].append(v)
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    str_keys = ['subject', 'view', 'category', 'nonfall_type', 'activity', 'group_id', 'orientation']
    np.savez_compressed(
        OUT_NPZ,
        X_joint=np.asarray(J, np.float32), X_motion=np.asarray(M, np.float32), X_angle=np.asarray(A, np.float32),
        **{k: np.asarray(v, dtype='U64' if k in str_keys else np.int64 if k == 'y' else np.int32)
           for k, v in meta.items()})
    print(f'-> {OUT_NPZ}  windows={len(J)}  videos dùng được={int((stats.n_windows > 0).sum())}/{len(stats)}')
    print(f'-> {OUT_CSV}')


def cmd_report(args):
    pd.set_option('display.width', 200)
    stats = pd.read_csv(OUT_CSV)
    stats['nonfall_type'] = stats['nonfall_type'].fillna('')
    stats['error'] = stats['error'].fillna('')
    print('Video bị loại:'); print(stats[stats.error != ''][['rel_path', 'n_frames', 'n_detected', 'error']].to_string())
    print('\nshape_mismatch =', int(stats.shape_mismatch.sum()))
    print('\nSố video / số cửa sổ theo người × nhãn × cam:')
    print(stats.pivot_table(index=['subject', 'category'], columns='view', values=['video_id', 'n_windows'],
                            aggfunc={'video_id': 'count', 'n_windows': 'sum'}, margins=True).to_string())
    print('\nHướng video × nhãn theo người:')
    print(pd.crosstab([stats.subject, stats.orientation], stats.label).to_string())
    print('\nTỷ lệ frame phát hiện được khung xương:')
    print(stats.groupby(['subject', 'view']).detected_ratio.describe()[['mean', 'min', '25%']].round(3).to_string())
    # overlay khung xương lên 1 frame: 1 video dọc + 1 video ngang
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    SANITY_DIR.mkdir(parents=True, exist_ok=True)
    vdf = collect_videos().set_index('video_id')
    for o in ['portrait', 'landscape']:
        s = stats[(stats.orientation == o) & (stats.n_windows > 0) & (stats.label == 1)]
        if not len(s):
            continue
        r = s.iloc[0]; v = vdf.loc[r.video_id]
        cache = pickle.load(open(cache_path(v.cache_key), 'rb'))
        fi = min(30, len(cache['skel']) - 1)
        cap = cv2.VideoCapture(v.path); cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, frame = cap.read(); cap.release()
        plt.figure(figsize=(6, 6 * frame.shape[0] / frame.shape[1]))
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); k = cache['skel'][fi]
        for a, b in EDGES:
            plt.plot([k[a, 0], k[b, 0]], [k[a, 1], k[b, 1]], 'lime', lw=2)
        plt.scatter(k[:, 0], k[:, 1], c='red', s=12); plt.axis('off'); plt.title(f'{o} {r.img_w}x{r.img_h}')
        plt.savefig(SANITY_DIR / f'overlay_{o}.png', dpi=60, bbox_inches='tight'); plt.close()
        print('->', SANITY_DIR / f'overlay_{o}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['extract', 'build', 'report'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 1)))
    a = ap.parse_args()
    print(f'mediapipe {mp.__version__} | opencv {cv2.__version__} | numpy {np.__version__} | python {sys.version.split()[0]}')
    {'extract': cmd_extract, 'build': cmd_build, 'report': cmd_report}[a.step](a)
