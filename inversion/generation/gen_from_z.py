# -*- coding: utf-8 -*-
"""
Created on Sat May 19 10:04:09 2018

@author: elaloy <elaloy elaloy@sckcen.be>
"""
import numpy as np
from scipy.signal import medfilt

def generate(generator,
             z,
             filtering=False,
             threshold=False):

    model = generator(z)
    model=model.detach().cpu().numpy()
    model = (model + 1) * 0.5  # Convert from [-1,1] to [0,1]

    # Postprocess if requested
    if filtering:
        for ii in range(model.shape[0]):
            model[ii, :] = medfilt(model[ii, 0,:,:], kernel_size=(3, 3))

    if threshold:
        threshold = 0.5
        model[model < threshold] = 0
        model[model >= threshold] = 1

    return model

if __name__ == "__main__":
    generate()
