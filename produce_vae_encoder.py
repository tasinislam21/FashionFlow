import torchvision.transforms as transforms
import os.path as osp
import cv2
import torch
import os
import tqdm
from PIL import Image
from diffusers import AutoencoderKL
import random
device = torch.device("cuda:4")

vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="vae",
            revision="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c"
        ).to(device)
vae.requires_grad_(False)

@torch.no_grad()
def VAE_encode(video):
    for i in range(video.shape[0]):
        image = video[i, :, :, :]
        image = image.unsqueeze(0)
        if i == 0:
            init_latent_dist = vae.encode(image).latent_dist.sample()
            init_latent_dist *= 0.18215
            encoded_video = (init_latent_dist).unsqueeze(1)
        else:
            init_latent_dist = vae.encode(image).latent_dist.sample()
            init_latent_dist *= 0.18215
            encoded_video = torch.cat([encoded_video, (init_latent_dist).unsqueeze(1)], 1)
    return encoded_video

def get_transform():
    image_transforms = transforms.Compose(
        [
        transforms.Resize((640, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        ])
    return image_transforms


path = osp.join("dataset_mp4")
video_names = os.listdir(path)
transform = get_transform()

for video_name in tqdm.tqdm(video_names):
    cap = cv2.VideoCapture(osp.join(path, video_name))
    numberOfFrames = 241
    number = random.randint(0, numberOfFrames - 70)
    for i in range(number, number + 70):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame)
        if i == number:
            inputImage = frame
            torch.save(inputImage, "dataset_vae/" + video_name[:-4] + "_image.pt")
            frame = frame.unsqueeze(0)
            restOfVideo = torch.clone(frame)
        else:
            frame = frame.unsqueeze(0)
            restOfVideo = torch.cat([restOfVideo, frame], 0)
    restOfVideo = restOfVideo.to(device=device)
    vae_video = VAE_encode(restOfVideo).detach().cpu()
    torch.save(vae_video, "dataset_vae/" + video_name[:-4] + ".pt")