import os
import sys
import torch
import random
import argparse
import numpy as np
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

import os
path = os.path.dirname(os.path.realpath(__file__))

def load_small_model(device):
    #print ("find free enouph device")
    
    print("text_generator_for_out", path)
    if os.path.exists(path + '/' + 'gpt2-pytorch_model.bin'):
        print ("exist1")
        state_dict = torch.load(path + '/' + 'gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
        print ("exist2")
        config = GPT2Config()
        print ("exist5")
        model = GPT2LMHeadModel(config)
        print ("exist6")
        model = load_weight(model, state_dict)
        print ("exist7")
        print (device)
        #torch.cuda.set_device(device)
        model.to(device)
        model.eval()
        return model
    else:
        raise RuntimeError("Can't load small gpt model")
    

def text_generator_for_out(text, model, device, length = 200, temperature = 0.7, top_k = 40, path_to_model = path):
    print("text_generator_for_out", path)
    if os.path.exists(path + '/' + 'gpt2-pytorch_model.bin'):
        print (path + '/' + 'gpt2-pytorch_model.bin')
        enc = get_encoder()
        quiet = False
        length = 200
        print ("text_generator_for_out 1")
        if length == -1:
            length = 1024 // 2
        elif length > 1024:
            raise ValueError("Can't get samples longer than window size: %s" % 1024)

        context_tokens = enc.encode(text)

        generated = 0
        print ("text_generator_for_out 2")
        for _ in range(1):
            out = sample_sequence(
                model=model, length=length,
                context=context_tokens,
                start_token=None,
                batch_size=1,
                temperature=temperature, top_k=top_k, device=device
            )
            print ("text_generator_for_out 3")
            out = out[:, len(context_tokens):].tolist()
            for i in range(1):
                generated += 1
                text = enc.decode(out[i])
                print ("text_generator_for_out 4")
                if quiet is False:
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print("in big gen2", text)
                return text