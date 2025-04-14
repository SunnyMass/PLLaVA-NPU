import torch
import time
import cv2
from PIL import Image
from argparse import ArgumentParser
import torchvision.transforms as transforms

from tasks.eval.eval_utils import conv_templates, ChatPllava
from tasks.eval.model_utils import load_pllava

SYSTEM = """You are a powerful Video Magic ChatBot, a large vision-language assistant. 
You are able to understand the video content that the user provides and assist the user in a video-language related task.
The user might provide you with the video and maybe some extra noisy information to help you out or ask you a question. Make use of the information in a proper way to be competent for the job.
### INSTRUCTIONS:
1. Follow the user's instruction.
2. Be critical yet believe in yourself.
"""
SYSTEM2 = """
Describe this video. Pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. Here are some examples of good descriptions: 1. A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about. 2. Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field. 3. Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff's edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.
"""

def load_video(video_path, num_segments=4, resolution=336):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total_frames / num_segments) for i in range(num_segments)]

    frames = []
    resize = transforms.Resize((resolution, resolution))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            pil_img = resize(pil_img)
            frames.append(pil_img)
        idx += 1
    cap.release()
    return frames


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--num_frames', type=int, default=4)
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True)
    parser.add_argument('--weight_dir', type=str, default=None)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_alpha', type=int, default=4)
    parser.add_argument('--conv_mode', type=str, default='plain')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--video_caption', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    print("📦 Loading model...")
    model, processor = load_pllava(
        repo_id=args.pretrained_model_name_or_path,
        num_frames=args.num_frames,
        use_lora=args.use_lora,
        lora_alpha=args.lora_alpha,
        weight_dir=args.weight_dir,
    )
    model = model.to('npu').eval()
    print(f"Model device: {next(model.parameters()).device}")
    chat = ChatPllava(model, processor)

    print("📽️ Loading video frames...")
    frames = load_video(args.video_path, args.num_frames)
    img_list = [frames]  # 必须是二维列表 [ [PIL, PIL, PIL...] ]

    print("💬 Asking and answering...")
    conv = conv_templates[args.conv_mode].copy()
    if args.video_caption:
        conv = chat.ask(args.prompt, conv, SYSTEM2)
    else:
        conv = chat.ask(args.prompt, conv, SYSTEM)

    start_time = time.time()
    llm_message, _, _ = chat.answer(
        conv=conv,
        img_list=img_list,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        temperature=args.temperature
    )
    elapsed = time.time() - start_time

    print(f"\n⏱️ Inference took {elapsed:.2f} seconds")
    print("\n===== FINAL ANSWER =====\n")
    print(llm_message.strip())


if __name__ == '__main__':
    main()
