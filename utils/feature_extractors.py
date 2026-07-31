#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

'''
    feature_extractors.py
    This module provides various feature extraction methods for different types of data.
'''

import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

def OneHotEncoding(element, set=None, extreme_value=1):
    # Provides a one-hot encoding of the element based on its position in the provided set.
    if isinstance(element, tuple):
        element = element[0]
    if element not in set:
        raise ValueError(f"Element '{element}' not found in the set.")

    index = set.index(element)
    encoding = np.zeros(len(set), dtype=int)
    encoding[index] = extreme_value
    return encoding

def IdentityEncoding(element):
    # Returns the element as a numpy array without any transformation. This is useful for cases where the raw value is already suitable for use as a feature.
    return np.array([element])

def FlattenEncoding(element):
    # Flattens the input element into a one-dimensional numpy array. If the input is a dictionary, it first converts it to a list of its values before flattening. This is useful for handling nested structures and ensuring that the output is a flat array suitable for machine learning models.
    if isinstance(element, dict):
        element = list(element.values())
    return np.array(element).flatten()

# Dictionaries used to cache loaded models and tokenizers to avoid redundant loading and improve performance when the same model is used multiple times.
_model_cache = {}
_tokenizer_cache = {}

def LanguageModelEmbedding(element, model_name, hf_key, language_mapping=None):
    # Generates an embedding for the input element using a specified language model. It first checks if the tokenizer and model for the given model name are already cached; if not, it loads them from Hugging Face. The input element can be optionally mapped to a different value using a provided language mapping. The function then tokenizes the input with truncation to fit the model's maximum length and computes the mean pooling of the token embeddings to produce a fixed-size vector representation of the input.
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_key)
    tokenizer = _tokenizer_cache[model_name]

    if model_name not in _model_cache:
        _model_cache[model_name] = AutoModel.from_pretrained(model_name, use_auth_token=hf_key)
    model = _model_cache[model_name]

    if language_mapping is not None:
        element = language_mapping.get(element[0], element)

    # Tokenize with truncation
    inputs = tokenizer(
        element,
        return_tensors='pt',
        truncation=True,    # truncate to model's max length
        max_length=tokenizer.model_max_length
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling over token embeddings
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

