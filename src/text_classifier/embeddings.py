from __future__ import annotations

from typing import Any, Iterable

from sentence_transformers import SentenceTransformer


def build_sentence_transformer(parameters: dict[str, Any]) -> SentenceTransformer:
    init_kwargs: dict[str, Any] = {}

    trust_remote_code = parameters.get("trust_remote_code")
    if trust_remote_code is not None:
        init_kwargs["trust_remote_code"] = bool(trust_remote_code)

    truncate_dim = parameters.get("truncate_dim")
    if truncate_dim is not None:
        init_kwargs["truncate_dim"] = int(truncate_dim)

    prompts = parameters.get("prompts")
    if prompts:
        init_kwargs["prompts"] = prompts

    default_prompt_name = parameters.get("default_prompt_name")
    if default_prompt_name:
        init_kwargs["default_prompt_name"] = default_prompt_name

    model_kwargs = parameters.get("model_kwargs")
    if model_kwargs:
        init_kwargs["model_kwargs"] = model_kwargs

    tokenizer_kwargs = parameters.get("tokenizer_kwargs")
    if tokenizer_kwargs:
        init_kwargs["tokenizer_kwargs"] = tokenizer_kwargs

    encoder = SentenceTransformer(
        parameters["name"],
        device=parameters["device"],
        **init_kwargs,
    )

    max_seq_length = parameters.get("max_seq_length")
    if max_seq_length is not None:
        encoder.max_seq_length = int(max_seq_length)

    return encoder


def encode_texts(
    encoder: SentenceTransformer, texts: Iterable[str], parameters: dict[str, Any]
):
    encode_kwargs: dict[str, Any] = {
        "batch_size": parameters["batch_size"],
        "normalize_embeddings": parameters["normalize_embeddings"],
        "show_progress_bar": False,
    }

    prompt = parameters.get("prompt")
    if prompt:
        encode_kwargs["prompt"] = prompt
    else:
        prompt_name = parameters.get("prompt_name")
        if prompt_name:
            encode_kwargs["prompt_name"] = prompt_name

    return encoder.encode(list(texts), **encode_kwargs)
