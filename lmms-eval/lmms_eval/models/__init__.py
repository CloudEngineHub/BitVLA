import importlib
import os
import sys

import hf_transfer
from loguru import logger

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stdout, level="WARNING")

AVAILABLE_MODELS = {
    "auroracap": "AuroraCap",
    "batch_gpt4": "BatchGPT4",
    "claude": "Claude",
    "cogvlm2": "CogVLM2",
    "from_log": "FromLog",
    "fuyu": "Fuyu",
    "gemini_api": "GeminiAPI",
    "gpt4v": "GPT4V",
    "idefics2": "Idefics2",
    "idefics3": "Idefics3",
    "instructblip": "InstructBLIP",
    "internvl": "InternVLChat",
    "internvl2": "InternVL2",
    "llama_vid": "LLaMAVid",
    "llava": "Llava",
    "moe_llava": "MoELlava",
    "llava_hf": "LlavaHf",
    "llava_onevision": "Llava_OneVision",
    "llava_onevision_moviechat": "Llava_OneVision_MovieChat",
    "llava_sglang": "LlavaSglang",
    "llava_vid": "LlavaVid",
    "slime": "Slime",
    "longva": "LongVA",
    "mantis": "Mantis",
    "minicpm_v": "MiniCPM_V",
    "minimonkey": "MiniMonkey",
    "moviechat": "MovieChat",
    "mplug_owl_video": "mplug_Owl",
    "phi3v": "Phi3v",
    "qwen_vl": "Qwen_VL",
    "qwen2_vl": "Qwen2_VL",
    "qwen2_5_vl": "Qwen2_5_VL",
    "qwen2_audio": "Qwen2_Audio",
    "qwen_vl_api": "Qwen_VL_API",
    "reka": "Reka",
    "srt_api": "SRT_API",
    "tinyllava": "TinyLlava",
    "videoChatGPT": "VideoChatGPT",
    "video_llava": "VideoLLaVA",
    "vila": "VILA",
    "xcomposer2_4KHD": "XComposer2_4KHD",
    "internvideo2": "InternVideo2",
    "xcomposer2d5": "XComposer2D5",
    "oryx": "Oryx",
    "videochat2": "VideoChat2",
    "llama_vision": "LlamaVision",
    "aria": "Aria",
}


def get_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")

    model_class = AVAILABLE_MODELS[model_name]
    if "." not in model_class:
        model_class = f"lmms_eval.models.{model_name}.{model_class}"

    try:
        model_module, model_class = model_class.rsplit(".", 1)
        module = __import__(model_module, fromlist=[model_class])
        return getattr(module, model_class)
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise


if os.environ.get("LMMS_EVAL_PLUGINS", None):
    # Allow specifying other packages to import models from
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        m = importlib.import_module(f"{plugin}.models")
        for model_name, model_class in getattr(m, "AVAILABLE_MODELS").items():
            AVAILABLE_MODELS[model_name] = f"{plugin}.models.{model_name}.{model_class}"
