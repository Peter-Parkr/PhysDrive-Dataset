#!/usr/bin/env python3
"""
generate_STMap_from_video.py

Convert each RGB.mp4 in On-Road-rPPG into STMap_RGB.png directly,
stored under each session's STMap/ folder.
Skips processing if STMap_RGB.png already exists.

Author: adapted from HSRD/BUAA implementation for PhysDrive
Supports: GPU acceleration via PyTorch (optional)
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import traceback
import shutil
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

try:
    import torch
    import torch.cuda.amp as amp
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ---------------- Parameters ----------------
ROOT = r"F:/autodl-tmp/PhysDrive"            # 📂 输入根目录
OUTPUT_ROOT = r"F:/autodl-tmp/PhysDrive-pre" # 📂 输出根目录
GRID_H, GRID_W = 5, 5                        # STMap 分块网格大小
GPU_BATCH = 256                              # 增大批处理大小（如果使用 GPU）
MIN_FRAMES = 10                              # 最小帧数阈值
OUTPUT_NAME = "STMap_RGB.png"                # 输出 STMap 文件名
NUM_WORKERS = 2                              # 预读取线程数（未使用，可忽略）
PREFETCH_SIZE = 4                            # 预读取队列大小（未使用，可忽略）
# --------------------------------------------


def uses_torch_cuda():
    return TORCH_AVAILABLE and torch.cuda.is_available()


def compute_block_means_torch(imgs_t, grid_h, grid_w):
    x = imgs_t.permute(0, 3, 1, 2)  # B,C,H,W
    B, C, H, W = x.shape
    h_step, w_step = H // grid_h, W // grid_w
    Hc, Wc = h_step * grid_h, w_step * grid_w
    x = x[:, :, :Hc, :Wc]
    x = x.view(B, C, grid_h, h_step, grid_w, w_step)
    x = x.mean(dim=3).mean(dim=4)
    x = x.permute(0, 2, 3, 1).reshape(B, grid_h * grid_w, C)
    return x


def compute_block_means_numpy(imgs_np, grid_h, grid_w):
    B, H, W, C = imgs_np.shape
    h_step, w_step = H // grid_h, W // grid_w
    imgs_np = imgs_np[:, :h_step * grid_h, :w_step * grid_w, :]
    imgs_np = imgs_np.reshape(B, grid_h, h_step, grid_w, w_step, C)
    imgs_np = imgs_np.mean(axis=2).mean(axis=3)
    return imgs_np.reshape(B, grid_h * grid_w, C)


def generate_stmap_from_video(video_path, out_path, use_gpu=True, batch_size=32):
    """优化后的 STMap 生成函数，使用批处理来减少内存占用"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < MIN_FRAMES:
        print(f"[WARN] {video_path} too short ({total} frames). Skip.")
        return False

    device = torch.device("cuda" if use_gpu and uses_torch_cuda() else "cpu")
    use_torch = TORCH_AVAILABLE
    
    # 使用列表存储每批次的结果
    all_means = []
    current_batch = []
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
                
            if frame is not None:
                current_batch.append(frame)
                
            # 当批次满了或达到视频末尾时处理当前批次
            if len(current_batch) >= batch_size or (not success and current_batch):
                batch = np.stack(current_batch).astype(np.float32)
                
                if use_torch:
                    imgs_t = torch.from_numpy(batch).to(device)
                    with torch.no_grad():
                        means = compute_block_means_torch(imgs_t, GRID_H, GRID_W).cpu().numpy()
                    del imgs_t
                    torch.cuda.empty_cache()  # 清理 GPU 缓存
                else:
                    means = compute_block_means_numpy(batch, GRID_H, GRID_W)
                
                all_means.append(means)
                current_batch = []  # 清空当前批次
                
            # 定期清理内存
            if len(all_means) % 10 == 0:
                torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"[ERROR] Processing {video_path}: {str(e)}")
        cap.release()
        return False
    finally:
        cap.release()

    try:
        # 合并所有批次的结果
        if not all_means:
            print(f"[ERROR] No valid frames processed in {video_path}")
            return False
            
        ST = np.concatenate(all_means, axis=0)
        T, NB, C = ST.shape
        STn = np.zeros_like(ST)
        
        # 逐通道处理以减少内存使用
        for c in range(C):
            for nb in range(NB):
                col = ST[:, nb, c]
                mn, mx = col.min(), col.max()
                STn[:, nb, c] = 0 if mx == mn else (col - mn) / (mx - mn + 1e-8)
        
        ST_img = (STn * 255).astype(np.uint8)
        ST_img = np.swapaxes(ST_img, 0, 1)
              
        # 确保输出目录存在
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, ST_img)
        
        # 清理内存
        del ST, STn, ST_img
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Saving STMap for {video_path}: {str(e)}")
        return False


def process_all(root):
    subjects = sorted(os.listdir(root))
    failed_videos = []
    
    for subj in tqdm(subjects, desc="Subjects", colour="cyan"):
        subj_path = os.path.join(root, subj)
        if not os.path.isdir(subj_path):
            continue
            
        sessions = sorted(os.listdir(subj_path))
        for sess in tqdm(sessions, desc=f"{subj}", leave=False, colour="green"):
            sess_path = os.path.join(subj_path, sess)
            if not os.path.isdir(sess_path):
                continue
            
            # 拷贝 Label 目录
            label_src = os.path.join(sess_path, "Label")
            if os.path.exists(label_src):
                label_dst = os.path.join(OUTPUT_ROOT, subj, sess, "Label")
                os.makedirs(label_dst, exist_ok=True)
                # 拷贝 Label 目录下的所有文件
                for label_file in os.listdir(label_src):
                    src_file = os.path.join(label_src, label_file)
                    dst_file = os.path.join(label_dst, label_file)
                    if not os.path.exists(dst_file):  # 如果目标文件不存在才复制
                        try:
                            shutil.copy2(src_file, dst_file)
                            tqdm.write(f"[COPY] {subj}/{sess}/Label/{label_file}")
                        except Exception as e:
                            tqdm.write(f"[ERROR] Failed to copy {src_file}: {str(e)}")
                
            # 处理视频文件
            video_path = os.path.join(sess_path, "Video", "RGB.mp4")
            out_path = os.path.join(OUTPUT_ROOT, subj, sess, "STMap", OUTPUT_NAME)
            
            if not os.path.exists(video_path):
                continue
            
            # ✅ 新增：检查 STMap 是否已经存在，如果存在则跳过
            if os.path.exists(out_path):
                tqdm.write(f"[SKIP] {subj}/{sess} -> STMap_RGB.png already exists, skipping.")
                continue
            
            try:
                # 使用较小的批处理大小
                ok = generate_stmap_from_video(video_path, out_path, batch_size=8)
                if ok:
                    tqdm.write(f"[OK] {subj}/{sess} -> STMap done.")
                else:
                    tqdm.write(f"[FAIL] {subj}/{sess}")
                    failed_videos.append(f"{subj}/{sess}")
            except Exception as e:
                tqdm.write(f"[ERR] {subj}/{sess}: {e}")
                failed_videos.append(f"{subj}/{sess}")
                traceback.print_exc()
            
            # 每处理完一个视频后清理内存
            torch.cuda.empty_cache()
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()
            
    # 打印失败的视频列表
    if failed_videos:
        print("\n⚠️ Failed videos:")
        for video in failed_videos:
            print(f"  - {video}")
        print(f"\nTotal failed: {len(failed_videos)}")


if __name__ == "__main__":
    print("=== Generating STMaps from Video (PhysDrive format) ===")
    print(f"Input path: {ROOT}")
    print(f"Output path: {OUTPUT_ROOT}")
    print(f"Using PyTorch: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 确保输出根目录存在
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 设置 OpenCV 的内存限制（可选）
    cv2.setNumThreads(1)  # 减少线程数
    
    try:
        process_all(ROOT)
        print("\n✅ Done! STMaps saved under:", OUTPUT_ROOT)
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        import sys
        sys.exit(1)