from separator.blocks.spleeternet import SpleeterNet

from config_files.General_config import General_config
from config_files.General_config import Train_config
from config_files.General_config import Separate_config
from config_files.General_config import Eval_config

HOME  = General_config["HOME"]
MODEL = General_config["model"]

from config_files.Model_config import Model_config
output_mask_logit = False
if Model_config["activition"] == "softmax":
    output_mask_logit = True

def create_model(phase):
    # if phase == "train":
    instrument_list =  Train_config["instrument_list"]
    # elif phase == "separate":
    #     instrument_list =  Separate_config["instrument_list"]
    # elif phase == "eval":
    #     instrument_list =  Eval_config["instrument_list"]

    if MODEL == "D3Net":
        from separator.blocks.d3net import D3Net, D3Net_
        sections = [Model_config['frequency_bins_low'], Model_config['frequency_bins'] - Model_config['frequency_bins_low']]

        in_channels = Model_config["in_channels"]
        num_features = Model_config["num_features"]
        growth_rate = Model_config["growth_rate"]
        bottleneck_channels = Model_config["bottleneck_channels"]

        kernel_size = Model_config['kernel_size']
        scale = Model_config['scale']
        depth = Model_config['depth']
        num_d3blocks  = Model_config["num_d3blocks"]
        num_d2blocks = Model_config["num_d2blocks"]

        kernel_size_d2block = Model_config["kernel_size_d2block"]
        growth_rate_d2block = Model_config["growth_rate_d2block"]
        depth_d2block = Model_config["depth_d2block"]

        kernel_size_gated = Model_config["kernel_size_gated"]

        model_ = D3Net_(
            in_channels, num_features, growth_rate, bottleneck_channels, kernel_size=kernel_size, sections=sections, scale=scale,
            num_d3blocks=num_d3blocks, num_d2blocks=num_d2blocks, depth=depth,
            growth_rate_d2block=growth_rate_d2block, kernel_size_d2block=kernel_size_d2block, depth_d2block=depth_d2block,
            kernel_size_gated=kernel_size_gated,phase ="train"
        )

        model = D3Net(model_,instrument_list, output_mask_logit, phase=phase)


    elif MODEL == "Spleeter":
        from separator.blocks.spleeternet import SpleeterNet
        model = SpleeterNet(instrument_list, output_mask_logit, phase=phase)

    elif MODEL == "ConvTasNet":
        from separator.blocks.conv_tasnet import ConvTasNet,ConvTasNet_
        batch_size = Model_config["batch_size"]
        C = Model_config["C"]
        T = Model_config["T"]
        L = Model_config["L"]
        stride = Model_config["stride"]
        N = Model_config["N"]
        H = Model_config["H"]
        B = Model_config["B"]
        Sc = Model_config["Sc"]
        P = Model_config["P"]
        R = Model_config["R"]
        X = Model_config["X"]
        sep_norm = Model_config["sep_norm"]
        enc_bases= Model_config["enc_bases"] 
        dec_bases = Model_config["dec_bases"]
        enc_nonlinear = Model_config["enc_nonlinear"]
        causal = Model_config["causal"]
        mask_nonlinear = Model_config["activition"]
        model_ = ConvTasNet_(N, kernel_size=L, stride=stride, enc_bases=enc_bases, dec_bases=dec_bases, enc_nonlinear=enc_nonlinear, sep_hidden_channels=H, sep_bottleneck_channels=B, sep_skip_channels=Sc, sep_kernel_size=P, sep_num_blocks=R, sep_num_layers=X, causal=causal, sep_norm=sep_norm, mask_nonlinear=mask_nonlinear)
        model = ConvTasNet(model_,instrument_list, phase=phase)

    else:
        print("Model is not specified.")
        from separator.blocks.spleeternet import SpsleeterNet
        model = SpleeterNet(instrument_list, output_mask_logit, phase=phase)
    return model