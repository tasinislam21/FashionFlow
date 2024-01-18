import pickle
import torch

with open('walter.pkl', 'rb') as f:
   data = pickle.load(f)

network = data['G'].cuda()

# with open('compressed_maybe.pkl', 'rb') as f:
#    data = pickle.load(f)
#
# network = data['Generator'].cuda()
all_z = torch.randn(1, 512, device='cuda')
ts = torch.arange(64, device='cuda').float().unsqueeze(0).repeat(1, 1)

all_c = torch.zeros(1, 0, device='cuda')
iters = range(len(all_z))

motion_z = network.synthesis.motion_encoder(c=all_c, t=ts)['motion_z']  # [...any...]
print(motion_z.shape) # [1, 86, 512]
out = network(
    z=all_z,
    c=all_c,
    t=ts,
    motion_z=motion_z,
    truncation_psi=1.0)
print(out.shape)
#torch.save(network.state_dict(), "face_weight.pth")
