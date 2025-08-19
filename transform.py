import json
import numpy as np
import cv2
import os
import logging
from tqdm import tqdm


class Affine(object):
    '''
    Make the spine of the skeleton perpendicular to the ground
    See the paper for a detailed description:https://arxiv.org/abs/2303.05234

    xx = x * cos - y * sin + x0 * (1 - cos) + y0 * sin
    yy = x * sin + y * cos + y0 * (1 - cos) - x0 * sin
    '''

    def __init__(self, fi=0):
        self.fi = fi

    def __call__(self, kp):
        kp_x = kp[..., 0]
        kp_y = kp[..., 1]
        neck_x = (kp_x[5] + kp_x[6]) / 2
        neck_y = (kp_y[5] + kp_y[6]) / 2
        hip_x = (kp_x[11] + kp_x[12]) / 2
        hip_y = (kp_y[11] + kp_y[12]) / 2

        x_l = hip_x - neck_x
        y_l = hip_y - neck_y
        theta = np.arctan(x_l / (y_l + 1e-6))  
        theta = np.expand_dims(theta, axis=-1)
        neck_x = np.expand_dims(neck_x, axis=-1)
        neck_y = np.expand_dims(neck_y, axis=-1)

        original_kp_x = kp[..., 0].copy()

        kp[..., 0] = np.cos(theta) * kp[..., 0] - np.sin(theta) * kp[..., 1] + (1 - np.cos(theta)) * neck_x + np.sin(theta) * neck_y
        kp[..., 1] = np.sin(theta) * original_kp_x + np.cos(theta) * kp[..., 1] + (1 - np.cos(theta)) * neck_y - np.sin(theta) * neck_x
        return kp, theta  

class RescaleCenter(object):
    def __init__(self, center='neck', scale=225):
        self.center = center
        self.scale = scale

    def __call__(self, kp):
        kp_x = kp[..., 0]
        kp_y = kp[..., 1]

        min_y = np.min(kp_y, axis=0)
        max_y = np.max(kp_y, axis=0)
        old_h = max_y - min_y
        new_h = self.scale
        projection = new_h / (old_h + 1e-6)  
        kp_x *= projection
        kp_y *= projection

        if self.center == 'neck':
            offset_x = (kp_x[5] + kp_x[6]) / 2
            offset_y = (kp_y[5] + kp_y[6]) / 2
        elif self.center == 'head':
            offset_x = (kp_x[1] + kp_x[2]) / 2
            offset_y = (kp_y[1] + kp_y[2]) / 2
        elif self.center == 'hip':
            offset_x = (kp_x[11] + kp_x[12]) / 2
            offset_y = (kp_y[11] + kp_y[12]) / 2

        kp_x -= offset_x
        kp_y -= offset_y
        return kp, offset_x, offset_y

class RescaleSilhouette(object):
    def __init__(self, scale=225, canvas_size=400, canvas_center_x=None, canvas_center_y=None):
        self.scale = scale
        self.canvas_size = canvas_size
        self.canvas_center_x = canvas_center_x if canvas_center_x is not None else canvas_size // 2
        self.canvas_center_y = canvas_center_y if canvas_center_y is not None else canvas_size // 3

    def __call__(self, contour_image, neck_x, neck_y):
        foreground_pixels = np.argwhere(contour_image == 255)

        if len(foreground_pixels) == 0:
            return np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)

        min_y = np.min(foreground_pixels[:, 0])
        max_y = np.max(foreground_pixels[:, 0])
        old_h = max_y - min_y
        new_h = self.scale
        projection = new_h / (old_h + 1e-6)  # Avoid dividing by zero

        
        scaling_matrix = np.array([[projection, 0, 0],
                                   [0, projection, 0]])
        scaled_contour = cv2.warpAffine(contour_image, scaling_matrix, (contour_image.shape[1], contour_image.shape[0]), flags=cv2.INTER_NEAREST)

        
        scaled_neck_x = neck_x * projection
        scaled_neck_y = neck_y * projection

        
        tx = self.canvas_center_x - scaled_neck_x
        ty = self.canvas_center_y - scaled_neck_y

        
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty]])
        silhouette_final = cv2.warpAffine(scaled_contour, translation_matrix, (self.canvas_size, self.canvas_size), flags=cv2.INTER_NEAREST, borderValue=0)

        return silhouette_final

def read_skeleton_txt(txt_path):
    try:
        with open(txt_path, 'r') as f:
            content = f.read()
        numbers = [float(num) for num in content.strip().split(',')]
        if len(numbers) < 2 + 17 * 3:
            raise ValueError("Insufficient number of numbers in skeleton data")
        width = numbers[0]
        height = numbers[1]
        keypoints = []
        scores = []
        for i in range(2, len(numbers), 3):
            x = numbers[i]
            y = numbers[i + 1]
            score = numbers[i + 2]
            keypoints.append([x, y])
            scores.append(score)
        keypoints = np.array(keypoints)
        scores = np.array(scores)
        return keypoints, scores
    except Exception as e:
        logging.error(f"Error reading skeleton txt file {txt_path}:{str(e)}")
        return None, None

def process_all_images(skeleton_root, silhouette_root, output_root, score_threshold=0.3):
    persons = os.listdir(skeleton_root)

    for person in tqdm(persons, desc="Processing persons", unit="person"):
        person_dir = os.path.join(skeleton_root, person)
        if not os.path.isdir(person_dir):
            continue

        for camid_videoid in os.listdir(person_dir):
            camid_videoid_dir = os.path.join(person_dir, camid_videoid)
            if not os.path.isdir(camid_videoid_dir):
                continue

            for seq_name in os.listdir(camid_videoid_dir):
                seq_dir = os.path.join(camid_videoid_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    continue

                for txt_file in os.listdir(seq_dir):
                    if not txt_file.endswith('.txt'):
                        continue
                    txt_path = os.path.join(seq_dir, txt_file)
                    silhouette_file = txt_file.replace('.txt', '.png')
                    silhouette_path = os.path.join(silhouette_root, person, camid_videoid, seq_name, silhouette_file)

                    if not os.path.exists(silhouette_path):
                        logging.warning(f"No silhouette found:{silhouette_path}")
                        continue

                    keypoints, scores = read_skeleton_txt(txt_path)
                    if keypoints is None:
                        continue

                    # Check confidence scores for shoulder and hip joints
                    if np.min(scores[5:7]) < score_threshold or np.min(scores[11:13]) < score_threshold:
                        logging.info(f"The file is skipped because of low confidence:{txt_path}")
                        continue

                    keypoints_copy = np.copy(keypoints)

                    # Apply affine transformation
                    affine_transform = Affine()
                    try:
                        transformed_keypoints, theta = affine_transform(keypoints_copy)
                    except Exception as e:
                        logging.error(f"Error performing affine transformation in file {txt_path}:{str(e)}")
                        continue

                    # Apply RescaleCenter transformation
                    rescale_center = RescaleCenter(center='neck', scale=225)
                    try:
                        transformed_keypoints, offset_x, offset_y = rescale_center(transformed_keypoints)
                    except Exception as e:
                        logging.error(f"Error performing RescaleCenter transformation in file {txt_path} :{str(e)}")
                        continue

                    # read silhouettes
                    contour_image = cv2.imread(silhouette_path, cv2.IMREAD_GRAYSCALE)
                    if contour_image is None:
                        logging.error(f"Error reading the silhouette image{silhouette_path}")
                        continue

                    # Affine trnsformation
                    neck_x_original = (keypoints[5, 0] + keypoints[6, 0]) / 2
                    neck_y_original = (keypoints[5, 1] + keypoints[6, 1]) / 2
                    M = cv2.getRotationMatrix2D((neck_x_original, neck_y_original), -np.degrees(float(theta[0])), 1)
                    transformed_contour = cv2.warpAffine(contour_image, M, (contour_image.shape[1], contour_image.shape[0]), borderValue=0)

                    # Extract the transformed neck coordinates
                    neck_coords = np.array([neck_x_original, neck_y_original, 1])
                    transformed_neck_coords = np.dot(M, neck_coords)
                    transformed_neck_x, transformed_neck_y = transformed_neck_coords[:2]

                    # Apply the RescaleSilhouette class for scaling and centralization
                    canvas_size = 400
                    canvas_center_x = canvas_size // 2
                    canvas_center_y = canvas_size // 3
                    rescale_silhouette = RescaleSilhouette(scale=225, canvas_size=canvas_size, canvas_center_x=canvas_center_x, canvas_center_y=canvas_center_y)
                    try:
                        silhouette_final = rescale_silhouette(transformed_contour, transformed_neck_x, transformed_neck_y)
                    except Exception as e:
                        logging.error(f"Error rescaling silhouette in file {txt_path}:{str(e)}")
                        continue

                    # Save the processed image
                    output_dir = os.path.join(output_root, person, camid_videoid, seq_name)
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = os.path.join(output_dir, f"{os.path.splitext(txt_file)[0]}.png")
                    cv2.imwrite(output_file, silhouette_final)
                    logging.info(f"Processed and saved:{output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Alignment")
    parser.add_argument('--skeleton_root', type=str, help='skeleton_root')
    parser.add_argument('--silhouette_root', type=str, help='silhouette_root')
    parser.add_argument('--output_root', type=str, help='image output root')
    parser.add_argument('--log_dir', type=str, default='./logs', help='log')
    parser.add_argument('--score_threshold', type=float, default=0.3, help='confidence threshold')

    args = parser.parse_args()

    # setting log
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.log_dir, 'transform_errors.log'),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    process_all_images(args.skeleton_root, args.silhouette_root, args.output_root, args.score_threshold)

