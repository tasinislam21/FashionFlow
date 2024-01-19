import os, argparse
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import random
import torch
import torch.utils.data as data
import os.path as osp
import cv2
import torchvision.transforms as transforms
import tqdm
from models.unet_dual_encoder import Embedding_Adapter
from models.diffusion_model import SpaceTimeUnet
import numpy as np
import torchvision.transforms.functional as TVF
from diffusers import AutoencoderKL
from PIL import Image
from transformers import CLIPVisionModel, CLIPProcessor
import torch.nn.functional as F

device = "cuda"
frameLimit = 70

parser = argparse.ArgumentParser(description="Configuration of the inference script.")
parser.add_argument("--pretrained_model", default="checkpoint/model_2300_global_and_local.pth", help="Path to a pretrained model")
parser.add_argument('--dataset', default="fashion_dataset/test", help="Path to a dataset")
parser.add_argument('--num_of_vid', type=int, default=1, help="Number of videos to be synthesised per conditional image")
args = parser.parse_args()

def cosine_beta_schedule(timesteps, start=0.0001, end=0.02):
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
betas = cosine_beta_schedule(timesteps=T)
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

class VideoFrameDataset(data.Dataset):
    def __init__(self):
        super(VideoFrameDataset, self).__init__()
        self.path = osp.join(args.dataset)
        self.video_names = os.listdir(self.path)
        self.transform = get_transform()

    def __getitem__(self, index):
        video_name = self.video_names[index]
        cap = cv2.VideoCapture(osp.join(self.path ,video_name))
        numberOfFrames = 241
        number = random.randint(0, numberOfFrames - frameLimit)
        cap.set(cv2.CAP_PROP_POS_FRAMES, number)
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = self.transform(frame)
        inputImage = frame
        return {'image': inputImage, 'name': video_name[:-4]}

    def __len__(self):
        return len(self.video_names)


vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4",
            subfolder="vae",
            revision="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c")
vae.to(device)
vae.requires_grad_(False)

with torch.no_grad():
    Net = SpaceTimeUnet(
        dim = 64,
        channels = 4,
        dim_mult = (1, 2, 4, 8),
        temporal_compression = (False, False, False, True),
        self_attns = (False, False, False, True),
        condition_on_timestep=True
    ).to(device)
adapter = Embedding_Adapter(input_nc=1280, output_nc=1280).to(device)

clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_encoder.requires_grad_(False)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

checkpoint = torch.load(args.pretrained_model)
Net.load_state_dict(checkpoint['net'])
adapter.load_state_dict(checkpoint['adapter'])
del checkpoint
torch.cuda.empty_cache()

train_dataset = VideoFrameDataset()
batch = 1
train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch,
            num_workers=0)

def save_video_frames_as_mp4(frames, fps, save_path):
    frame_h, frame_w = frames[0].shape[2:]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(save_path, fourcc, fps, (frame_w, frame_h))
    frames = frames[0]
    for frame in frames:
        frame = np.array(TVF.to_pil_image(frame))
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()

@torch.no_grad()
def VAE_encode(image):
    init_latent_dist = vae.encode(image).latent_dist.sample()
    init_latent_dist *= 0.18215
    encoded_image = (init_latent_dist).unsqueeze(1)
    return encoded_image

@torch.no_grad()
def VAE_decode(video, vae_net):
    decoded_video = None
    for i in range(video.shape[1]):
        image = video[:, i, :, :, :]
        image = 1 / 0.18215 * image
        image = vae_net.decode(image).sample
        image = image.clamp(0,1)
        if i == 0:
            decoded_video = image.unsqueeze(1)
        else:
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

def tensor2image(tensor):
    numpy_image = tensor[0].cpu().detach().numpy()
    rescaled_image = (numpy_image * 255).astype(np.uint8)
    pil_image = Image.fromarray(rescaled_image.transpose(1, 2, 0))
    return pil_image

@torch.no_grad()
def get_image_embedding(input_image):
    inputs = clip_processor(images=list(input_image), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    clip_hidden_states = clip_encoder(**inputs).last_hidden_state.to(device)
    vae_hidden_states = vae.encode(input_image).latent_dist.sample() * 0.18215
    encoder_hidden_states = adapter(clip_hidden_states, vae_hidden_states)
    return encoder_hidden_states

if not os.path.exists("result/"):
    os.makedirs("result/")

step = 0
for data in train_dataloader:
    torch.cuda.empty_cache()
    step += 1
    name = data["name"][0]
    image = data['image'].to(device=device)
    encoder_hidden_states = get_image_embedding(input_image=image)
    img = tensor2image(image)
    img.save("result/" + name + ".jpg")
    encoded_image = VAE_encode(image)
    for video_num in range(args.num_of_vid):
        noise_video = torch.randn([1, frameLimit, 4, 80, 64]).to(device)
        noise_video[:, 0:1] = encoded_image
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, T)[::-1]):
                t = torch.full((1,), i, device=device).long()
                noise_video = sample_timestep(noise_video, encoder_hidden_states, t)
                noise_video[:, 0:1] = encoded_image
            final_video = VAE_decode(noise_video, vae)
        save_video_frames_as_mp4(final_video, 25, "result/" + name + str(video_num) + ".mp4")