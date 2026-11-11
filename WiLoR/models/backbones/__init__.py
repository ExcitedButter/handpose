from .vit import vit


def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vit':
        return vit(cfg)
    raise NotImplementedError(f"Backbone type is not implemented in the dependency-light port: {cfg.MODEL.BACKBONE.TYPE}")
