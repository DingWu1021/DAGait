import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import argparse

T_W = 44
T_H = 64

def cut_img(img):
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_AREA)
    
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    if x_center < 0:
        return None
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right]
    return img.astype('uint8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gait3D dataset Preprocessing.')
    parser.add_argument('--sil_path', default='/data/wzx/Gait3D_transform', type=str,
                        help='Root path of transformed silhouette dataset.')
    parser.add_argument('-o', '--output_path', default='/data/wzx/Gait3D_transform_pkl',
                        type=str, help='Output path of pickled dataset.')
    args = parser.parse_args()

    for person_id in tqdm(sorted(os.listdir(args.sil_path)), desc='Processing persons'):
        person_dir = os.path.join(args.sil_path, person_id)
        if not os.path.isdir(person_dir):
            continue
        for camid_videoid in sorted(os.listdir(person_dir)):
            camid_videoid_dir = os.path.join(person_dir, camid_videoid)
            if not os.path.isdir(camid_videoid_dir):
                continue
            for seq_name in sorted(os.listdir(camid_videoid_dir)):
                seq_dir = os.path.join(camid_videoid_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    continue
                aligned_segs = []
                for seg_file in sorted(os.listdir(seq_dir)):
                    seg_path = os.path.join(seq_dir, seg_file)
                    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
                    if seg is None:
                        print("Not Found or Invalid Image: " + seg_path)
                        continue
                    
                    aligned_seg = cut_img(seg)
                    if aligned_seg is not None:
                        aligned_segs.append(aligned_seg)
                if len(aligned_segs) > 0:
                    output_path = os.path.join(
                        args.output_path, person_id, camid_videoid, seq_name)
                    os.makedirs(output_path, exist_ok=True)
                    
                    output_file = os.path.join(
                        output_path, f"{seq_name}-aligned-sils.pkl")
                    with open(output_file, 'wb') as f:
                        pickle.dump(np.asarray(aligned_segs), f)
                else:
                    print("No valid aligned silhouettes found: " +
                          seq_dir)
                    continue


