"""
When loading the model from the saved state_dict file, remove the prefix
"""
def strip_state_prefix(state_dict, custom_prefix="_orig_mod.module."):
    ddp_prefix = "_orig_mod.module."
    regular_prefix = "_orig_mod."

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(ddp_prefix):
            new_key = k[len(ddp_prefix):]
        elif k.startswith(regular_prefix):
            new_key = k[len(regular_prefix):]
        elif k.startswith(custom_prefix):
            new_key = k[len(custom_prefix):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict