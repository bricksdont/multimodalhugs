#!/usr/bin/env python
# coding=utf-8
"""
Script to generate predictions using model.generate().

This version removes evaluation/metrics entirely and does not use Trainer.
It keeps:
- CLI/YAML argument loading
- logging/telemetry setup
- dataset loading + preprocessing transform
- model/processor/collator setup
- batched generation and writing predictions to disk
"""

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    HfArgumentParser,
    set_seed,
    GenerationConfig,
)
from multimodalhugs.processors import (
    SignwritingProcessor,
    Pose2TextTranslationProcessor,
    Image2TextTranslationProcessor,
    Text2TextTranslationProcessor,
    Features2TextTranslationProcessor,
)

import multimodalhugs.models

Pose2TextTranslationProcessor.register_for_auto_class()
AutoProcessor.register("pose2text_translation_processor", Pose2TextTranslationProcessor)

Features2TextTranslationProcessor.register_for_auto_class()
AutoProcessor.register("features2text_translation_processor", Features2TextTranslationProcessor)

SignwritingProcessor.register_for_auto_class()
AutoProcessor.register("signwritting_processor", SignwritingProcessor)

Image2TextTranslationProcessor.register_for_auto_class()
AutoProcessor.register("image2text_translation_processor", Image2TextTranslationProcessor)

Text2TextTranslationProcessor.register_for_auto_class()
AutoProcessor.register("text2text_translation_processor", Text2TextTranslationProcessor)

import logging
import sys
import json
import torch
import datasets
import transformers

from pathlib import Path
from typing import Union
from transformers.utils import send_example_telemetry

from multimodalhugs.utils import print_module_details

from multimodalhugs.tasks.translation.config_classes import (
    ModelArguments,
    ProcessorArguments,
    DataTrainingArguments,
    ExtraArguments,
    ExtendedSeq2SeqTrainingArguments,
    GenerateArguments,
)

from multimodalhugs.tasks.translation.inference_utils import batched_inference, ModalityType

from multimodalhugs.tasks.translation.utils import (
    construct_kwargs,
    merge_config_and_command_args,
    resolve_missing_arg,
    resolve_checkpoint_path_from_general_setup_path,
)

logger = logging.getLogger(__name__)


def save_dict_to_json(data: dict, path: Union[str,Path], indent: int = 2) -> None:
    """
    Save a dictionary of strings to a JSON file.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def _infer_generation_kwargs(generate_args, generation_config):
    """
    Keep behavior similar to translation_generate.py:
    - If generate_args.max_length is set, treat it as "how many new tokens to generate".
    - Use num_beams when provided.
    """
    gen_kwargs = {}

    if getattr(generate_args, "num_beams", None) is not None:
        gen_kwargs["num_beams"] = generate_args.num_beams

    if getattr(generate_args, "max_length", None) is not None:
        # Most intuitive for "generate continuation"
        gen_kwargs["max_new_tokens"] = generate_args.max_length

    if generation_config is not None:
        gen_kwargs["generation_config"] = generation_config

    return gen_kwargs


def main():
    # --- Argument parsing (CLI + optional YAML merge) ---
    parser = HfArgumentParser(
        (GenerateArguments, ExtraArguments, ModelArguments, ProcessorArguments, DataTrainingArguments, ExtendedSeq2SeqTrainingArguments)
    )
    generate_args, extra_args, model_args, processor_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if extra_args.config_path:
        for section in ("training",):
            try:
                training_args = merge_config_and_command_args(
                    extra_args.config_path,
                    ExtendedSeq2SeqTrainingArguments,
                    section,
                    training_args,
                    sys.argv[1:],
                )
                break
            except KeyError:
                continue
        generate_args = merge_config_and_command_args(extra_args.config_path, GenerateArguments, "generation", generate_args, sys.argv[1:])
        model_args = merge_config_and_command_args(extra_args.config_path, ModelArguments, "model", model_args, sys.argv[1:])
        processor_args = merge_config_and_command_args(extra_args.config_path, ProcessorArguments, "processor", processor_args, sys.argv[1:])
        data_args = merge_config_and_command_args(extra_args.config_path, DataTrainingArguments, "data", data_args, sys.argv[1:])

    # Ensure we don't drop columns needed by multimodal processors/collators
    setattr(training_args, "remove_unused_columns", False)
    setattr(training_args, "report_to", [])

    # Resolve paths if omitted
    if model_args.model_name_or_path is None:
        resolve_missing_arg(
            model_args,
            "model_name_or_path",
            training_args.output_dir,
            extra_args.setup_path if hasattr(extra_args, "setup_path") else None,
        )
        model_args.model_name_or_path = resolve_checkpoint_path_from_general_setup_path(model_args.model_name_or_path)

    resolve_missing_arg(
        processor_args,
        "processor_name_or_path",
        training_args.output_dir,
        extra_args.setup_path if hasattr(extra_args, "setup_path") else None,
    )

    # Telemetry (optional)
    send_example_telemetry("run_translate_only", model_args, data_args)

    # --- Logging ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # argument handling

    if generate_args.input is None:
        raise ValueError("Must specify an input TSV file with --input.")
    if generate_args.output is None:
        raise ValueError("Must specify an output JSON file for predictions with --output.")

    logger.info(f"*** Translating input file {generate_args.input} ***")

    # --- Seed ---
    set_seed(training_args.seed)

    # --- Config + generation config ---
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Same behaviour as translation_generate.py: max_length arg maps to max_new_tokens
    if generate_args.max_length is not None:
        config.max_new_tokens = generate_args.max_length
        config.max_length = None
    elif hasattr(config, "max_new_tokens") and config.max_new_tokens is not None:
        config.max_length = None

    generation_config = GenerationConfig.from_model_config(config)

    # --- Processor / tokenizer ---
    if not processor_args.processor_name_or_path:
        raise ValueError("You must specify processor_name_or_path in the config or on the command line")

    processor_kwargs = construct_kwargs(processor_args, ["processor_name_or_path"])
    processor = AutoProcessor.from_pretrained(processor_args.processor_name_or_path, **processor_kwargs)
    for key in set(processor_kwargs.keys()):
        if hasattr(processor, key):
            setattr(processor, key, processor_kwargs.pop(key))
    tokenizer = processor.tokenizer

    # --- Model ---
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    logger.info(f"\n{model}\n")
    logger.info(f"\n{print_module_details(model)}\n")

    # --- collator arguments ---
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    pad_to_multiple_of = 8 if training_args.fp16 else None

    # --- dataloader arguments ---
    per_device_bs = getattr(training_args, "per_device_eval_batch_size", None) or 8

    # --- Device / precision ---
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    gen_kwargs = _infer_generation_kwargs(generate_args, generation_config)

    # --- Generate ---

    modality: ModalityType = "pose2text"

    result = batched_inference(model=model,
                               processor=processor,
                               tsv_path=generate_args.input,
                               modality=modality,
                               batch_size=per_device_bs,
                               pad_to_multiple_of=pad_to_multiple_of,
                               label_pad_token_id=label_pad_token_id,
                               gen_kwargs=gen_kwargs)

    # Save predictions as JSON

    logger.info(f"Saving translations to: {generate_args.output}")
    save_dict_to_json(data=result, path=generate_args.output)


if __name__ == "__main__":
    main()
