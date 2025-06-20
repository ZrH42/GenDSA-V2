import os
import glob
import argparse
import shutil
import re
from tqdm import tqdm
from Simple_Interpolator import run_interpolator
from Trainer import Model
import config as cfg
from PIL import Image
import cv2

def sorted_nicely(l):
    return sorted(l, key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)])

def extract_number_from_filename(filename):
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    numbers = re.findall(r'\d+', name_without_ext)
    if numbers:
        return int(numbers[0])
    return None

def resize_image_if_large(image_path, max_size=1024):
    try:
        img = cv2.imread(image_path)
        if img is None:
            with Image.open(image_path) as img_pil:
                if img_pil.width > max_size or img_pil.height > max_size:
                    img_pil.thumbnail((max_size, max_size), Image.LANCZOS)
                    img_pil.save(image_path)
        else:
            h, w = img.shape[:2]
            if w > max_size or h > max_size:
                if w > h:
                    new_w, new_h = max_size, int(h * max_size / w)
                else:
                    new_w, new_h = int(w * max_size / h), max_size
                img = cv2.resize(img, (new_w, new_h))
                cv2.imwrite(image_path, img)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def process_directory(input_folder, output_folder, model, inter_frames):
    for root, dirs, files in os.walk(input_folder):
        rename_images_in_directory(root)
        
        frame_files = sorted_nicely(glob.glob(os.path.join(root, '*.png')))
        
        if frame_files:
            relative_path = os.path.relpath(root, input_folder)
            current_output_folder = os.path.join(output_folder, relative_path)
            os.makedirs(current_output_folder, exist_ok=True)
            
            interpolate_sequence(root, current_output_folder, model, inter_frames)

def rename_images_in_directory(directory):
    frame_files = sorted_nicely(glob.glob(os.path.join(directory, '*.png')))
    
    for frame in frame_files:
        basename = os.path.basename(frame)
        numbers = re.findall(r'\d+', basename)
        if numbers:
            new_name = f"{numbers[0]}.png"
            new_path = os.path.join(directory, new_name)
            
            if basename != new_name and not os.path.exists(new_path):
                try:
                    os.rename(frame, new_path)
                    print(f"Renamed: {basename} -> {new_name}")
                except Exception as e:
                    print(f"Rename failed {basename}: {e}")

def interpolate_sequence(input_folder, output_folder, model, inter_frames):
    frame_files = sorted_nicely(glob.glob(os.path.join(input_folder, '*.png')))
    num_frames = len(frame_files)

    selected_frames = [frame_files[i] for i in range(num_frames) if i % (inter_frames + 1) == 0 and i >= 0]

    for frame in selected_frames:
        resize_image_if_large(frame)
        dest_path = os.path.join(output_folder, os.path.basename(frame))
        shutil.copy(frame, dest_path)

    for i in tqdm(range(len(selected_frames) - 1), desc="Processing frames"):
        Frame1 = selected_frames[i]
        Frame2 = selected_frames[i + 1]
        
        TimeStepList = [(j + 1) / (inter_frames + 1) for j in range(inter_frames)]
        
        Output_Frames_list = []
        for j in range(inter_frames):
            frame1_num = int(os.path.splitext(os.path.basename(Frame1))[0])
            frame2_num = int(os.path.splitext(os.path.basename(Frame2))[0])
            
            step = (frame2_num - frame1_num) / (inter_frames + 1)
            insert_index = int(frame1_num + step * (j + 1))
            Output_Frames_list.append(os.path.join(output_folder, f'{insert_index}_InterFrame.png'))
        
        run_interpolator(model, Frame1, Frame2, time_list=TimeStepList, Output_Frames_list=Output_Frames_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='/path/to/input/dir', help='Input frames directory path')
    parser.add_argument('--output_folder', type=str, default='/path/to/output/dir', help='Output interpolated frames directory path')
    parser.add_argument('--model_path', type=str, default='model', help='Model path')
    parser.add_argument('--inter_frames', type=int, default=3, help='Number of interpolation frames')
    args = parser.parse_args()

    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=32,
        lambda_range='local',
        depth=[2, 2, 2, 4]
    )
    model = Model(-1)
    model.load_model(full_path=args.model_path)
    model.eval()
    model.device()

    process_directory(args.input_folder, args.output_folder, model, args.inter_frames)