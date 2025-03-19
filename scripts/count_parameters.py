from source.network.sky_voice_net import SkyVoiceNet
from source.utilities import count_parameters

net = SkyVoiceNet(mode='double_attn')

count_parameters(net)
