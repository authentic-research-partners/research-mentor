"""Vision model for image artifact understanding.

Uses Qwen3-VL-2B-Instruct to generate text descriptions of images when
text extraction produces no text (pure images: photos, diagrams, charts, handwritten notes).

Always runs on CPU — must not compete with vLLM for GPU VRAM.
CPU inference is slow (~15-30s per image) but acceptable at upload time.

Same lazy-singleton pattern as embeddings.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Lazy singleton — loaded on first use
_model: Qwen3VLForConditionalGeneration | None = None
_processor: AutoProcessor | None = None


def _get_model() -> tuple[Qwen3VLForConditionalGeneration, Any]:
    """Get or create the singleton vision model and processor.

    Always loads on CPU with float32 to avoid competing with vLLM for GPU VRAM.
    ~4GB RAM at 2B params.
    """
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from research_mentor.config import load_config

    config = load_config()
    model_name = config.vision.model

    logger.info("Loading vision model: {} (device=cpu, dtype=float32)", model_name)
    _processor = AutoProcessor.from_pretrained(model_name)  # type: ignore[no-untyped-call]
    _model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    logger.info("Vision model loaded: {}", model_name)
    return _model, _processor


def describe_image(file_path: str | Path) -> str | None:
    """Generate a text description of an image.

    Args:
        file_path: Path to the image file.

    Returns:
        Text description of the image, or None if the file doesn't exist
        or description fails.
    """
    path = Path(file_path)
    if not path.is_file():
        logger.warning("describe_image: file not found: {}", path)
        return None

    from research_mentor.config import load_config

    config = load_config()

    model, processor = _get_model()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(path)},
                {
                    "type": "text",
                    "text": (
                        "Examine this image carefully. "
                        "First, transcribe any text you can read "
                        "(printed, typed, or handwritten). "
                        "Then describe any diagrams, charts, graphs, tables, "
                        "or visual elements present in the image."
                    ),
                },
            ],
        },
    ]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_input],
        images=[path],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cpu")

    import torch

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=config.vision.max_new_tokens)

    # Decode only the generated tokens (skip the input)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    description: str = processor.decode(generated_ids, skip_special_tokens=True).strip()

    if not description:
        return None

    logger.info("Generated image description ({} chars) for {}", len(description), path.name)
    return description


def reset_model() -> None:
    """Reset the singleton model (for tests)."""
    global _model, _processor
    _model = None
    _processor = None
