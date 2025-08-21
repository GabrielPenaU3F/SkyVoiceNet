
import torch.nn as nn

from source.data_processing.normalizer import Normalizer


class MinMaxWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, speech_spec, melody_contour, target_spec=None):
        # Min-max normalize each
        speech_spec_norm, _, _ = Normalizer.minmax_normalize_batch(speech_spec)

        if self.training:
            # En entrenamiento también normalizamos el target
            assert target_spec is not None, "target_spec is required during training"
            target_norm, _, _ = Normalizer.minmax_normalize_batch(target_spec)
            output_norm = self.model(speech_spec_norm, melody_contour)
            return output_norm, target_norm
        else:
            # En inferencia NO se necesita target, solo desnormalizamos con la escala del input
            output_norm = self.model(speech_spec_norm, melody_contour)

            # Desnormalizamos usando los mínimos y máximos del target real
            # Pero como no lo tenemos, usamos el input como proxy
            # (alternativamente, podrías guardar stats globales de target al entrenar)
            speech_mins = speech_spec.amin(dim=(1, 2), keepdim=True)
            speech_maxs = speech_spec.amax(dim=(1, 2), keepdim=True)
            output = Normalizer.minmax_denormalize(output_norm, speech_mins, speech_maxs)
            # return output
            return output_norm