from .nodes import ACE15_AudioDecoder

NODE_CLASS_MAPPINGS = {
    "ACE15_AudioDecoder": ACE15_AudioDecoder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE15_AudioDecoder": "ACE15 Audio Decoder (Tiled)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]