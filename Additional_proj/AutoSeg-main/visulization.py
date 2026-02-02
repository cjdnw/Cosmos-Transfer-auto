import numpy as np
from PIL import Image, ImageEnhance
import os
from tqdm import tqdm
import cv2
import random
import argparse
import imageio


def images_to_video(input_folder, output_video, fps=30,max_width=1920, max_height=1080):
    images = [img for img in os.listdir(input_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))  

    if not images:
        print("No images found in the folder.")
        return

    frames = [imageio.imread(os.path.join(input_folder, img)) for img in images]
    imageio.mimwrite(output_video, frames, fps=30)
    

def load_masks_from_png(mask_png_path, image_size):
    """
    Recover per-object masks from the concatenated PNG saved by auto-mask-batch.py.
    The PNG width is (image_width * num_objects), height is image_height.
    """
    image_width, image_height = image_size
    mask_image = Image.open(mask_png_path).convert("L")
    mask_array = np.array(mask_image)
    height, width = mask_array.shape

    if height != image_height or width % image_width != 0:
        raise ValueError(
            f"Mask PNG size mismatch: got ({width}, {height}), expected "
            f"width multiple of {image_width} and height {image_height}."
        )

    num_objects = width // image_width
    masks = []
    for obj_idx in range(num_objects):
        start = obj_idx * image_width
        end = (obj_idx + 1) * image_width
        mask_slice = mask_array[:, start:end]
        masks.append((mask_slice > 0).astype(np.uint8))

    masks = np.array(masks)[:, np.newaxis, :, :]
    return masks


def load_masks_for_frame(base_dir, frame_id, image_size, prefer_npy=True):
    """
    Load masks for a frame.
    - Prefer NPY (mask_XXX.npy) when present.
    - Fallback to concatenated PNG (mask_XXX.png) to support --save_npy 0.
    """
    npy_path = os.path.join(base_dir, f"mask_{frame_id:03}.npy")
    png_path = os.path.join(base_dir, f"mask_{frame_id:03}.png")

    if prefer_npy and os.path.exists(npy_path):
        masks = np.load(npy_path)
        if masks.ndim == 3:
            masks = masks[:, np.newaxis, :, :]
        if masks.ndim != 4:
            raise ValueError(f"Unexpected mask array shape {masks.shape} in {npy_path}")
        return masks

    if os.path.exists(png_path):
        return load_masks_from_png(png_path, image_size)

    raise FileNotFoundError(
        f"Mask file for frame {frame_id:03} not found. Checked {npy_path} and {png_path}."
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",type=str,required=True)
    parser.add_argument("--output_dir",type=str,required=True)
    parser.add_argument("--level",choices=['default','small','middle','large'])
    parser.add_argument("--enable_highlight",type=int,default=1,help="是否生成单对象高亮/掩码帧与视频（1 生成，0 不生成）")
    parser.add_argument("--enable_overlay",type=int,default=1,help="是否生成彩色叠加视频（1 生成，0 不生成）")
    parser.add_argument(
        "--overlay_style",
        type=str,
        default="blend",
        choices=["blend","mask"],
        help="彩色叠加样式：blend=在原图上半透明叠加；mask=仅彩色块（不含原图）。仅当 enable_overlay=1 时生效。"
    )
    args = parser.parse_args()

    level = args.level
    basedir = args.output_dir
    npybasedir = os.path.join(basedir,level,'final-output')
    imagebasedir = args.video_path
    highlight_savedir = os.path.join(basedir,level,'visualization','seperate')
    enable_highlight = bool(args.enable_highlight)
    enable_overlay = bool(args.enable_overlay)
    if enable_highlight:
        os.makedirs(highlight_savedir,exist_ok=True)
    image_name_list = os.listdir(imagebasedir)
    image_name_list.sort()

    npy_name_list = []
    for name in os.listdir(npybasedir):
        if name.endswith(".npy"):
            npy_name_list.append(name)
    npy_name_list.sort()
    print(npy_name_list)

    image_list = [Image.open(os.path.join(imagebasedir,name)) for name in image_name_list]
    prefer_npy = len(npy_name_list) > 0
    mask_seq = []
    for frame_id, image in enumerate(image_list):
        masks = load_masks_for_frame(npybasedir, frame_id, image.size, prefer_npy=prefer_npy)
        mask_seq.append(masks)

    print(len(mask_seq))

    if enable_highlight:
        uid_dirs = {}
        for uid in range(mask_seq[0].shape[0]):
            uid_dir = os.path.join(highlight_savedir, f"uid_{uid}")
            os.makedirs(uid_dir, exist_ok=True)
            uid_dirs[uid] = uid_dir

        for frame_id, (masks, image) in tqdm(enumerate(zip(mask_seq,image_list))):
            dark_image = ImageEnhance.Brightness(image).enhance(0.3)
            dark_image_array = np.array(dark_image)
            highlighted_images = []
            mask_images = []
            for uid in range(masks.shape[0]):

                current_mask = masks[uid][0]
                image_array = np.array(image)

                highlighted_image_array = np.where(current_mask[:, :, np.newaxis], image_array, dark_image_array)
                highlighted_image = Image.fromarray(highlighted_image_array.astype('uint8'))

                bool_mask_array = (current_mask * 255).astype(np.uint8)
                bool_mask_image = Image.fromarray(bool_mask_array).convert("RGB")

                uid_frame = Image.new('RGB', (image.width, image.height * 2))
                uid_frame.paste(highlighted_image, (0, 0))
                uid_frame.paste(bool_mask_image, (0, image.height))
                uid_frame.save(os.path.join(uid_dirs[uid], f"{frame_id:05}.png"))

                highlighted_images.append(highlighted_image)
                mask_images.append(bool_mask_image)

            total_width = image.width * len(highlighted_images)
            max_height = image.height * 2

            final_image = Image.new('RGB', (total_width, max_height))

            for i, img in enumerate(highlighted_images):
                final_image.paste(img, (i * image.width, 0))

            for i, img in enumerate(mask_images):
                final_image.paste(img, (i * image.width, image.height))

            final_image.save(os.path.join(highlight_savedir,f"{frame_id:05}.png"))

    overlay_savedir = os.path.join(basedir,level,'visualization','full-mask-npy')
    video_path = os.path.join(basedir,level,'visualization',f'full-mask-{level}.mp4')
    if enable_overlay:
        os.makedirs(overlay_savedir,exist_ok=True)
    image_name_list = os.listdir(imagebasedir)
    image_name_list.sort()
    print(image_name_list)
    print(npy_name_list)
    image_list = [Image.open(os.path.join(imagebasedir,name)) for name in image_name_list]
    assert len(mask_seq) == len(image_list), "Number of masks does not match number of images."
    print(len(mask_seq))
    if enable_overlay:
        # Generate a list of random colors
        def generate_random_colors(num_colors):
            colors = []
            for _ in range(num_colors):
                color = tuple(random.randint(0, 255) for _ in range(3))
                colors.append(color)
            return colors

        num_masks = max(len(masks) for masks in mask_seq)
        colors = generate_random_colors(num_masks)

        video_frames = []
        output_path_list = []
        for frame_id, (masks, image) in tqdm(enumerate(zip(mask_seq, image_list))):
            image_np = np.array(image)
            mask_combined = np.zeros_like(image_np, dtype=np.uint8)

            for i, mask in enumerate(masks):
                color = colors[i % len(colors)]
                # Ensure the mask is binary (0 or 1)
                mask_binary = (mask[0] > 0).astype(np.uint8)
                # Apply the mask color
                for j in range(3):  # For each channel in RGB
                    mask_combined[:, :, j] += mask_binary * color[j]

            mask_combined = np.clip(mask_combined, 0, 255)
            if args.overlay_style == "blend":
                # Blend masks with the original image
                blended_image = cv2.addWeighted(image_np, 0.7, mask_combined, 0.3, 0)
            else:
                # Pure color masks on black background
                blended_image = mask_combined

            # Save each frame
            output_path = os.path.join(overlay_savedir, f"frame_{frame_id:04d}.png")
            output_path_list.append(output_path)

            Image.fromarray(blended_image).save(output_path)
            
        # Generate the video

        frames = [imageio.imread(img) for img in output_path_list]
        imageio.mimwrite(video_path, frames, fps=30)

        print(f"Video saved at {video_path}")
    else:
        print("Skip overlay generation as enable_overlay=0")

    if enable_highlight:
        output_video_dir = os.path.join(basedir,level,'visualization','seperate','videos')
        os.makedirs(output_video_dir,exist_ok=True)

        for uid in tqdm(range(len(uid_dirs))):
            uid_folder = uid_dirs[uid]
            images_to_video(uid_folder, os.path.join(output_video_dir,f"uid_{uid}.mp4"))
