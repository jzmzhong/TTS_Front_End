import os
import torch
from thop import profile
from thop import clever_format

import sys
sys.path.append("../../")
sys.path.append("../")

from dp.model.model import load_checkpoint
from dp.model.predictor import Predictor

MODELS = ["1_baseline/autoreg_EnUs_layer4+4_dim512_ffn4_head8",
        #   "1_baseline/autoreg_EnUs_layer6+4_dim512_ffn4_head8",
          # "3_forward_trimmed/forward_EnUs_random106k_layer4_dim512_ffn4_head8",
          # "3_forward_trimmed/forward_EnUs_random106k_layer3_dim384_ffn2_head6",
          # "3_forward_trimmed/forward_trimmed_EnUs_random106k_layer4_dim512_ffn4_head8",
          # "3_forward_trimmed/forward_trimmed_EnUs_random106k_layer3_dim384_ffn2_head6",
          ]

for MODEL in MODELS:
    # model & its path
    print(MODEL)
    checkpoint_path = "../checkpoints/{}/latest_model.pt".format(MODEL)

    # load model
    model, checkpoint = load_checkpoint(checkpoint_path, device="cuda")
    preprocessor = checkpoint["preprocessor"]
    predictor = Predictor(model, preprocessor)

    # prepare inputs
    # INPUT_LENGTH = 7 * 3 + 2
    # INPUT_LENGTH = 20 * 3 + 2
    # INPUT_LENGTH = 7 * 1 + 2
    INPUT_LENGTH = 20 * 1 + 2
    text = torch.randint(1, 20, (1, INPUT_LENGTH))
    text_len = torch.tensor([INPUT_LENGTH])
    start_index = torch.tensor([1])
    # phonemes = torch.randint(1, 50, (1, 7 + 2))
    phonemes = torch.randint(1, 50, (1, 20 + 2))
    inputs = {'text': text, 'text_len': text_len, 'start_index': start_index, 'phonemes': phonemes}

    # get macs & params
    # print(sum([v.numel() for k, v in model.state_dict().items()]))
    macs, params = profile(model, inputs=(inputs,))
    macs, params = clever_format([macs, params], "%.3f")
    print(params)
    print(macs)