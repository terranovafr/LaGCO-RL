#!/usr/bin/env python3
# Copyright (c) 2026 Franco Terranova.
# This file is licensed under the GNU General Public License v3.0.
# You may redistribute it and/or modify it under the terms of the GPL-3.0.
# See the LICENSE file in the project root for the full license text.

'''
    encoding_options.py
    This module defines the encoding options for different types of values enabled by our framework.
'''


class ContinuousValue:
    def __init__(self, normalization=None, mean=None, std=None, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        self.normalization = normalization
        self.mean = mean
        self.std = std

class BinaryValue:
    pass

class RankingValue:
    def __init__(self, normalization=None, mean=None, std=None, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        self.normalization = normalization
        self.mean = mean
        self.std = std

class RankingContinuousValue:
    def __init__(self, normalization=None, mean=None, std=None, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        self.normalization = normalization
        self.mean = mean
        self.std = std

class MultiClassValue:
    def __init__(self, num_classes):
        self.num_classes = num_classes