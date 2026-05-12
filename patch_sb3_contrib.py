# patch_sb3_contrib.py
# Fixes the error "TypeError: __init__() got an unexpected keyword argument 'validate_args'" in sb3_contrib's MaskableCategorical distribution by patching the __init__ method to remove or ignore the validate_args argument when it is passed. This allows the code to run without modification while still ensuring that the distribution is initialized correctly without the unsupported argument.

import sb3_contrib.common.maskable.distributions as dist

# Save original __init__
_original_init = dist.Categorical.__init__

def patched_init(self, *args, **kwargs):
    # Remove validate_args if present in kwargs
    if 'validate_args' in kwargs:
        kwargs.pop('validate_args')

    # Replace or append validate_args=False in the correct positional place
    # MaskedCategorical signature: __init__(self, probs=None, logits=None, validate_args=None)
    # So we pass args[0] -> probs, args[1] -> logits, then validate_args=False
    if len(args) == 0:
        # no probs or logits passed
        _original_init(self, validate_args=False, **kwargs)
    elif len(args) == 1:
        # only probs
        _original_init(self, args[0], validate_args=False, **kwargs)
    elif len(args) == 2:
        # probs and logits
        _original_init(self, args[0], args[1], validate_args=False, **kwargs)
    else:
        # too many args? fallback to original
        _original_init(self, *args, **kwargs)


# Replace original with patched
dist.Categorical.__init__ = patched_init
