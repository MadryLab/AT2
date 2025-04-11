def is_multimodal(model_name):
    if "gemma-3-" in model_name.lower():
        params_start = model_name.lower().find("gemma-3-") + len("gemma-3-")
        params_end = model_name.lower().find("b-it", params_start)
        params = model_name[params_start: params_end]
        # Only gemma-3-1b-it is text-only, the rest are multimodal
        if params != "1":
            return True
    return False