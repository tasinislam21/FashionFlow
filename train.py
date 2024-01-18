import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
from transformers import CLIPVisionModel, CLIPProcessor
import torch.nn as nn
import torch
import torch.utils.data as data
import os.path as osp
import cv2
import torchvision.transforms as transforms
import torch.optim as optim
from tensorboardX import SummaryWriter
import argparse
import numpy as np
import torchvision.transforms.functional as TVF
import torch.nn.functional as F
from models.unet_dual_encoder import Embedding_Adapter
from distributed import (get_rank, synchronize)
from diffusers import AutoencoderKL
from models.diffusion_model import SpaceTimeUnet

parser = argparse.ArgumentParser(description="Pose with Style trainer")
parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
args = parser.parse_args()

torch.distributed.init_process_group(backend="nccl", init_method="env://")
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)
synchronize()

frameLimit = 70

if get_rank() == 0:
    writer = SummaryWriter('samevideo')

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    betas = []
    for i in reversed(range(timesteps)):
        T = timesteps - 1
        beta = start + 0.5 * (end - start) * (1 + np.cos((i / T) * np.pi))
        betas.append(beta)
    return torch.Tensor(betas)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

T = 1000
betas = linear_beta_schedule(timesteps=T)
# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def get_transform():
    image_transforms = transforms.Compose(
        [
        transforms.Resize((640, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        ])
    return image_transforms

# class VideoFrameDataset(data.Dataset):
#     def __init__(self):
#         super(VideoFrameDataset, self).__init__()
#         self.path = osp.join("dataset_mp4")
#         self.video_names = os.listdir(self.path)
#         self.transform = get_transform()
#
#     def __getitem__(self, index):
#         video_name = self.video_names[index]
#         cap = cv2.VideoCapture(osp.join(self.path ,video_name))
#         numberOfFrames = 241
#         number = random.randint(0, numberOfFrames - frameLimit)
#         for i in range(number, number + frameLimit):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#             _, frame = cap.read()
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = Image.fromarray(frame)
#             frame = self.transform(frame)
#             if i == number:
#                 inputImage = frame
#             frame = frame.unsqueeze(0)
#             if i == number:
#                 restOfVideo = torch.clone(frame)
#             else:
#                 restOfVideo = torch.cat([restOfVideo, frame], 0)
#         return {'image': inputImage, 'video': restOfVideo}
#
#     def __len__(self):
#         return len(self.video_names)

class VideoFrameDataset(data.Dataset):
    def __init__(self):
        super(VideoFrameDataset, self).__init__()
        self.path = osp.join("dataset_mp4")
        self.vae_path = osp.join("dataset_vae")
        self.video_names = os.listdir(self.path)
        self.transform = get_transform()

    def __getitem__(self, index):
        video_name = self.video_names[index]
        inputImage = torch.load(osp.join(self.vae_path, video_name[:-4]+"_image.pt"), map_location='cpu')
        restOfVideo = torch.load(osp.join(self.vae_path, video_name[:-4]+".pt"), map_location='cpu')[0]
        return {'image': inputImage, 'video': restOfVideo}

    def __len__(self):
        return len(self.video_names)

vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="vae",
            revision="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c"
        ).to(device)
vae.requires_grad_(False)

@torch.no_grad()
def VAE_encode(image):
    init_latent_dist = vae.encode(image).latent_dist.sample()
    init_latent_dist *= 0.18215
    encoded_image = (init_latent_dist).unsqueeze(1)
    return encoded_image

Net = SpaceTimeUnet(
    dim = 64,
    channels = 4,
    dim_mult = (1, 2, 4, 8),
    temporal_compression = (False, False, False, True),
    self_attns = (False, False, False, True),
    condition_on_timestep = True,
).to(device)
adapter = Embedding_Adapter(input_nc=1280, output_nc=1280).to(device)

clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
clip_encoder.requires_grad_(False)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


parameters = list(Net.parameters()) + list(adapter.parameters())
optimizerG = optim.AdamW(parameters, lr=0.0001, weight_decay=0.01)

Net = nn.parallel.DistributedDataParallel(
        Net,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False)

adapter = nn.parallel.DistributedDataParallel(
        adapter,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False)

checkpoint = torch.load('checkpoint/vae_clip_1300_162625.pth')
Net_state_dict = {}
for k, v in checkpoint['net'].items():
    name = 'module.' + k  # add the prefix 'module'
    Net_state_dict[name] = v
Net.load_state_dict(Net_state_dict)

adapter_state_dict = {}
for k, v in checkpoint['adapter'].items():
    name = 'module.' + k  # add the prefix 'module'
    adapter_state_dict[name] = v
adapter.load_state_dict(adapter_state_dict)
optimizerG.load_state_dict(checkpoint['opt'])
del checkpoint
del Net_state_dict
del adapter_state_dict
torch.cuda.empty_cache()

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

train_dataset = VideoFrameDataset()
sampler = data_sampler(train_dataset, shuffle=True, distributed=True)
batch = 2
train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch,
            sampler=sampler,
            num_workers=1,
            drop_last=True)

def save_video_frames_as_mp4(frames, fps, save_path):
    frame_h, frame_w = frames[0].shape[2:]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(save_path, fourcc, fps, (frame_w, frame_h))
    frames = frames[0]
    for frame in frames:
        #assert frame.shape[0] == 3, "RGBA/grayscale images are not supported"
        frame = np.array(TVF.to_pil_image(frame))
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()

mseloss = torch.nn.MSELoss(reduction="mean")


def get_loss(input_image, latent_video):
    timesteps = torch.randint(0, T, (batch,), device=device)
    timesteps = timesteps.long()
    initial_frame_latent_video = latent_video[:, 0:1].clone().detach() # [b, f, c, h, w]
    x_noisy, noise = forward_diffusion_sample(latent_video, timesteps)
    x_noisy[:, 0:1] = initial_frame_latent_video
    noise[:, 0:1] = torch.zeros(initial_frame_latent_video.shape)
    x_noisy = x_noisy.permute(0, 2, 1, 3, 4)

    inputs = clip_processor(images=list(input_image), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    clip_hidden_states = clip_encoder(**inputs).last_hidden_state.to(device)
    vae_hidden_states = vae.encode(input_image).latent_dist.sample() * 0.18215
    encoder_hidden_states = adapter(clip_hidden_states, vae_hidden_states)

    noise_pred = Net(x_noisy, encoder_hidden_states, timestep=timesteps.float())
    noise_pred = noise_pred.permute(0, 2, 1, 3, 4)
    loss = 0.0
    for i in range(frameLimit):
        loss += mseloss(noise_pred[:, i, :, :, :], noise[:, i, :, :, :])
    return loss

# @torch.no_grad()
# def VAE_encode(video):
#     for i in range(video.shape[1]):
#         image = video[:, i, :, :, :]
#         if i == 0:
#             init_latent_dist = vae.encode(image).latent_dist.sample()
#             init_latent_dist *= 0.18215
#             encoded_video = (init_latent_dist).unsqueeze(1)
#         else:
#             init_latent_dist = vae.encode(image).latent_dist.sample()
#             init_latent_dist *= 0.18215
#             encoded_video = torch.cat([encoded_video, (init_latent_dist).unsqueeze(1)], 1)
#     return encoded_video

@torch.no_grad()
def VAE_decode(video):
    decoded_video = None
    for i in range(video.shape[1]):
        image = video[:, i, :, :, :]
        image = 1 / 0.18215 * image
        if i == 0:
            image = vae.decode(image).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            decoded_video = image.unsqueeze(1)
        else:
            image = vae.decode(image).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            decoded_video = torch.cat([decoded_video, image.unsqueeze(1)], 1)
    return decoded_video


@torch.no_grad()
def sample_timestep(x, image, t):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    with torch.cuda.amp.autocast():
        sample_output = Net(x.permute(0, 2, 1, 3, 4), image, timestep=t.float())
    sample_output = sample_output.permute(0, 2, 1, 3, 4)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * sample_output / sqrt_one_minus_alphas_cumprod_t
    )
    if t.item() == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def get_image_embedding(input_image):
    inputs = clip_processor(images=list(input_image), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    clip_hidden_states = clip_encoder(**inputs).last_hidden_state.to(device)
    vae_hidden_states = vae.encode(input_image).latent_dist.sample() * 0.18215
    encoder_hidden_states = adapter(clip_hidden_states, vae_hidden_states)
    return encoder_hidden_states

step = 162625

for epoch in range(1301, 5000):
    Net.train()
    adapter.train()
    for data in train_dataloader:
        step += 1
        vae_video = data['video'].to(device=device) # [b, f, c, h, w]
        image = data['image'].to(device=device)

        loss = get_loss(input_image=image, latent_video=vae_video)
        optimizerG.zero_grad()
        loss.backward()
        optimizerG.step()

    if get_rank() == 0 and epoch % 40 == 0:
        writer.add_scalar('loss', loss, step)
    if get_rank() == 0 and epoch % 100 == 0:
        torch.save(
            {
                'net': Net.module.state_dict(),
                'adapter': adapter.module.state_dict(),
                'opt': optimizerG.state_dict()
            }, "checkpoint/vae_clip_" + str(epoch) + "_" + str(step) + ".pth")
    if get_rank() == 0 and epoch % 100 == 0:
        noise_video = torch.randn([1, frameLimit, 4, 80, 64]).to(device)
        encoder_hidden_states = get_image_embedding(input_image=image[0].unsqueeze(0))
        encoded_image = VAE_encode(image[0].unsqueeze(0))
        noise_video[:, 0:1] = encoded_image
        with torch.no_grad():
            for i in range(0, T)[::-1]:
                t = torch.full((1,), i, device=device).long()
                noise_video = sample_timestep(noise_video, encoder_hidden_states, t)
                noise_video[:, 0:1] = encoded_image
            final_video = VAE_decode(noise_video)
        writer.add_image('input image', image[0], step)
        writer.add_video('video', final_video, step)
        save_video_frames_as_mp4(final_video, 25, "sample/video"+str(epoch)+".mp4")

if get_rank() == 0:
    torch.save({
                'net': Net.module.state_dict(),
                'adapter': adapter.module.state_dict()
                }, "checkpoint/vae_clip_e100.pth")