"""
When loading the model from the saved state_dict file, remove the prefix
"""
def strip_state_prefix(state_dict, prefix="_orig_mod.module."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict