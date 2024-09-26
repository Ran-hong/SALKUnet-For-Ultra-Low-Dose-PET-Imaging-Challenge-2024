from models.Unet3D import UNet3d
from models.Unet3D_AKAB import UNet3d_AKAB
from models.RepLKUnet3D import RepLKUnet3d
from models.Unet3D_deep import UNet3d_deep

def modelFactory(config):
    model = config.model
    if model.name == "Unet3d":
        return UNet3d(
            in_channels = model.Unet3d.in_channels,
            out_channels = model.Unet3d.out_channels,
            init_features = model.Unet3d.init_features,
        )
    elif model.name == "Unet3d_AKAB":
        return UNet3d_AKAB(
            in_channels = model.Unet3d.in_channels,
            out_channels = model.Unet3d.out_channels,
            init_features = model.Unet3d.init_features,
        )
    elif model.name == "RepLKUnet3d":
        return RepLKUnet3d(
            in_channels=model.RepLKUnet.in_channels,
            out_channels=model.RepLKUnet.out_channels,
            mlp_ratio=model.RepLKUnet.mlp_ratio,
            features=model.RepLKUnet.features,
            lkSize=model.RepLKUnet.lkSize,
            skSize=model.RepLKUnet.skSize,
        )
    elif model.name == "Unet3d_deep":
        return UNet3d_deep(
            in_channels = model.Unet3d_deep.in_channels,
            out_channels = model.Unet3d_deep.out_channels,
            init_features = model.Unet3d_deep.init_features,
        )
    else:
        raise f"The model name {model.name} not exist!"