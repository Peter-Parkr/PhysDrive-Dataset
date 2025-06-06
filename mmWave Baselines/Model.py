from VResNet import ResNet3D_Radar
from STFormer import STFormer
from VUNet import UNet3D_Radar
from mmFormer.mmFormer import UNet_Transformer_BatchNorm_Attn_Cut_KD_MyTF_Multi_Expert
from VitaNet import VitaNet
from IQ_MVED import IQ_MVED

def My_model(model_name):
    if model_name == "ResNet":
        return ResNet3D_Radar()
    elif model_name == "STFormer":
        return STFormer(model_type='factorised_encoder')
    elif model_name == "UNet":
        return UNet3D_Radar()
    elif model_name == "mmFormer":
        return UNet_Transformer_BatchNorm_Attn_Cut_KD_MyTF_Multi_Expert(num_range_bins=8, num_doppler=8)
    elif model_name == "VitaNet":
        return VitaNet(num_frames=200, num_range_bins=8, num_doppler=8, num_angles=16)

    elif model_name == "IQ_MVED":
        return IQ_MVED(num_doppler=8)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")