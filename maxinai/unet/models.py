"""
Initialize UNet models
"""

from fastai2.vision.all import *
from fastai2.vision.models import unet, resnet34

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _add_norm(dls, meta, pretrained):
    if not pretrained: return
    after_batch = dls.after_batch
    if first(o for o in after_batch.fs if isinstance(o, Normalize)): return
    stats = meta.get('stats')
    if stats is None: return
    after_batch.add(Normalize.from_stats(*stats))


def init_model(weights: Path, arch: callable = resnet34, pretrained: bool = True, n_in: int = 3,
               n_out: int = 32, normalize: bool = True, **kwargs) -> nn.Module:
    """
    Initialize UNet model and load weights
    Args:
        weights: weights file path

    Returns:
        model: UNet model with trained weights
    """
    config = unet_config()
    meta = model_meta.get(arch, _default_meta)
    body = create_body(arch, n_in, pretrained, -2)
    size = torch.Size([96, 128])
    assert n_out, "`n_out` is not defined, and could not be infered from data, set `dls.c` or pass `n_out`"
    if normalize: _add_norm(dls, meta, pretrained)
    Normalize.from_stats(*imagenet_stats)
    model = unet.DynamicUnet(body, n_out, size, **config)

    return model
