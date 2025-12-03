import os
import argparse
from config import METHODS
from pipeline import StableDiffusionGenerator, AnimateDiffGenerator
import imageio.v2 as imageio
#from utils import compute_clipscore_torchmetrics
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import numpy as np
import torch
def parse_args():
    parser = argparse.ArgumentParser(description="A program for creating greenback images using diverse techniques")
    parser.add_argument("--method", type=str, choices=METHODS, default="tkg", help="Choose the generation technique (e.g., gbp, tkg)")
    parser.add_argument("--device", type=int, default=0, help="Index of the CUDA GPU to be used")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for random number generation to ensure reproducibility")
    parser.add_argument("--video", action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    device = f"cuda:{args.device}"
    
    """
    base_prompts = [
        "young woman with virtual reality glasses sitting in armchair",
        "yellow lemon and slice",
        "gray cat british shorthair",
        "vintage golden trumpet making music concept",
        "set of many business people"
    ]
    """
    base_prompts = [
        "A horse is running",
        "A cat is dancing",
        "a person is walking",
        "a girl is waving her hand"
    ]
    active_prompts = ', realistic, photo-realistic, 4K, high resolution, high quality'
    negative_prompts = 'background, character, cartoon, anime, text, fail, low resolution'

    if args.video:
        generator = AnimateDiffGenerator(method=args.method, device=device, seed=args.seed)

        prompts, generated_images = generator.generate_videos(
            base_prompts=base_prompts,
            active_prompts=active_prompts,
            negative_prompts=negative_prompts,
            num_frames = 32,
        )
        
        output_dir = os.path.join("outputs_video", "sdxl", args.method, str(args.seed))
        os.makedirs(output_dir, exist_ok=True)
        print("Output directory:", output_dir)

        ## Evluation and save
        clip_score_fn = partial(
            clip_score,
            model_name_or_path="openai/clip-vit-base-patch16"
        )

        clip_scores = []

        for prompt, frames in zip(prompts, generated_images):
            filename = f"{prompt.replace(' ', '_')[:]}.mp4"
            filepath = os.path.join(output_dir, filename)
            imageio.mimsave(
                filepath,
                frames,
                fps=8,
                quality=8,
                codec="libx264",
            )

            arr = np.stack(frames, axis=0).astype("uint8")
            print("arr.shape: ",arr.shape)
            img_tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to("cuda")

            # prompt 要重複成與 frames 相同長度
            prompt_list = [prompt] * img_tensor.size(0)

            score = clip_score_fn(img_tensor, prompt_list)
            score = float(score.detach().cpu())

            clip_scores.append(score)

            print(f"{filename}: CLIPScore = {score:.4f}")
        """""
        scores, mean_score = compute_clipscore_torchmetrics(
            video_dir=output_dir,
            prompts=prompts,
            num_frames=8,
            device="cuda"
        )
        """""
        print("Average CLIPScore:", sum(clip_scores) / len(clip_scores))
    else:
        generator = StableDiffusionGenerator(method=args.method, device=device, seed=args.seed)
        prompts, generated_images = generator.generate_images(
            base_prompts=base_prompts,
            active_prompts=active_prompts,
            negative_prompts=negative_prompts,
        )
        
        output_dir = os.path.join("outputs", "sdxl", args.method, str(args.seed))
        os.makedirs(output_dir, exist_ok=True)
        print("Output directory:", output_dir)

        for prompt, img in zip(prompts, generated_images):
            filename = f"{prompt.replace(' ', '_')[:10]}.png"
            img.save(os.path.join(output_dir, filename))

if __name__ == "__main__":
    main()
    
