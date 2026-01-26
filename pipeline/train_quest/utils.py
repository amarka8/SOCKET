# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import random, torch, os
import numpy as np


class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        self.__dict__ = dictionary


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _strip_special_tokens(text: str, tokenizer) -> str:
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for special_tok in special_token:
                text = text.replace(special_tok, "")
        else:
            text = text.replace(special_token, "")
    return text.strip()
