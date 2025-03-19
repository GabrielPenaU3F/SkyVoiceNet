from source.network.sky_voice_net import SkyVoiceNet
from source.utilities import count_parameters

net = SkyVoiceNet(mode='cat_pre_post_attn')

count_parameters(net)
