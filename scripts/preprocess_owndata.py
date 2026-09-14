import os
import sys

import cv2
from tqdm import tqdm


def resize_image(frame, max_size):
    if max_size <= 0:
        return frame

    h, w = frame.shape[:2]

    if max(h, w) <= max_size and h % 8 == 0 and w % 8 == 0:
        return frame

    scale = min(1.0, max_size / max(h, w))

    new_w = int(w * scale)
    new_h = int(h * scale)

    # Make dimensions multiples of 8
    new_w = max(8, (new_w // 8) * 8)
    new_h = max(8, (new_h // 8) * 8)

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_frames(input_video_path, output, fps, max_size):
    os.makedirs(output, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Couldn't open video file.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_step = max(1, int(round(video_fps / fps)))

    frame_number = 0
    saved_frame_number = 0

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                desc="Extracting frames")
    while True:
        # Read frame from video
        ret, frame = cap.read()

        # Break if no more frames are available
        if not ret:
            break

        if frame_number % frame_step == 0:
            frame = resize_image(frame, max_size)

            # Save the current frame as an image
            cv2.imwrite(os.path.join(
                output, f"{saved_frame_number:06d}.jpg"), frame)

            saved_frame_number += 1

        frame_number += 1
        pbar.update(1)

    # Release the video capture object
    pbar.close()
    cap.release()


def run_colmap(output):
    colmap_binary = "colmap"

    db = f"{output}/colmap.db"
    images = f"{output}/images"

    os.system(f"{colmap_binary} feature_extractor --ImageReader.camera_model OPENCV --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path {db} --image_path {images}")

    os.system(
        f"{colmap_binary} sequential_matcher --SiftMatching.guided_matching=true --database_path {db}")

    os.system(f"mkdir {output}/sparse")
    os.system(
        f"{colmap_binary} mapper --database_path {db} --image_path {images} --output_path {output}/sparse")

    os.system(f"{colmap_binary} bundle_adjuster --input_path {output}/sparse/0 --output_path {output}/sparse/0 --BundleAdjustment.refine_principal_point 1")

    os.system(f"mkdir {output}/sparse_txt")
    os.system(f"{colmap_binary} model_converter --input_path {output}/sparse/0 --output_path {output}/sparse_txt --output_type TXT")


# Example usage:
input_video_path = sys.argv[1]       # Path to the input video file
output = sys.argv[2]                 # Path to the output folder
fps = float(sys.argv[3])             # Target FPS
max_size = int(sys.argv[4])          # Maximum image side

extract_frames(
    input_video_path,
    output + "/images",
    fps,
    max_size
)

# Run colmap
run_colmap(output)

# dump calib file
calib = open(f'{output}/sparse_txt/cameras.txt').readlines()[-1].split()[4:]
open(f'{output}/calib.txt', 'w').write(" ".join(calib))
