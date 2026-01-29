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
    set_seed,
    GenerationConfig,
)

# noinspection PyUnresolvedReferences
import multimodalhugs.models

import logging
import json
import torch

from pathlib import Path
from typing import Union, Dict, List, Any, Optional
from transformers.utils import send_example_telemetry

from multimodalhugs.utils import print_module_details
from multimodalhugs.models.multimodal_embedder import MultiModalEmbedderModel, MultiModalEmbedderConfig
from multimodalhugs.tasks.translation.inference_utils import batched_inference, ModalityType
from multimodalhugs.tasks.translation.config_classes import GenerateArguments
from multimodalhugs.processors import MultimodalSequence2SequenceProcessor

from multimodalhugs.tasks.translation.utils import (
    construct_kwargs,
    set_up_logging,
    assemble_generation_args,
    register_processor_autoclasses
)

register_processor_autoclasses()

logger = logging.getLogger(__name__)


def save_dict_to_json(data: dict, path: Union[str,Path], indent: int = 2) -> None:
    """
    Save a dictionary of strings to a JSON file.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def _infer_generation_kwargs(generate_args: GenerateArguments):
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
        gen_kwargs["max_length"] = None

    return gen_kwargs


TranslationOutput = Dict[str, List[Any]]


class Translator:

    """
    Class that generates predictions using a provided model and processor.

    If components such as model, config and processor are not provided as objects, they will be initialized from scratch.
    Specifically,
    - if a model is provided, then the arguments config_name or model_name_or_path will be ignored.
    - if model is None, then config_name or model_name_or_path must be provided.

    - if a processor is provided, then the argument processor_name_or_path will be ignored.
    - if processor is None, then processor_name_or_path must be provided.
    """

    def __init__(self,
                 config: Optional[MultiModalEmbedderConfig] = None,
                 config_name: Optional[str] = None,
                 model_name_or_path: Optional[str] = None,
                 model: Optional[MultiModalEmbedderModel] = None,
                 processor: Optional[MultimodalSequence2SequenceProcessor] = None,
                 processor_name_or_path: Optional[str] = None,
                 processor_kwargs: dict = None,
                 modality: ModalityType = "pose2text",
                 batch_size: int = 8,
                 gen_kwargs: dict = None,
                 generation_config: GenerationConfig = None,
                 seed: Optional[int] = None,
                 cache_dir: Optional[str] = None,
                 model_revision: str = "main",
                 token: Optional[str] = None,
                 trust_remote_code: bool = False,
                 fp16: bool = False,
                 ignore_pad_token_for_loss: bool = True,
                 device: torch.device = torch.device("cpu")):

        self.config = config
        self.config_name = config_name
        self.model_name_or_path = model_name_or_path
        self.model = model
        self.processor = processor
        self.processor_name_or_path = processor_name_or_path
        self.processor_kwargs = processor_kwargs if processor_kwargs is not None else {}
        self.modality = modality
        self.batch_size = batch_size
        self.gen_kwargs = gen_kwargs if gen_kwargs is not None else {}
        self.generation_config = generation_config
        self.seed = seed
        self.cache_dir = cache_dir
        self.model_revision = model_revision
        self.token = token
        self.trust_remote_code = trust_remote_code
        self.fp16 = fp16
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.device = device

        if self.seed is not None:
            set_seed(self.seed)

        if self.model is None:
            if self.config_name is None and self.model_name_or_path is None:
                raise ValueError("If `model` is not provided, `config_name` or `model_name_or_path` must be provided to initialize the model.")

            # if model is not given, try to initialize config and model

            self.initialize_model()

        if self.generation_config is None:
            if self.config is None:
                raise ValueError("If `generation_config` is not provided, `config` must be provided to create a generation config.")

            # after initializing the model, config must exist as well

            self.generation_config = GenerationConfig.from_model_config(self.config)

        if self.processor is None:
            if self.processor_name_or_path is None:
                raise ValueError("If `processor` is not provided, `processor_name_or_path` must be provided to initialize the processor.")

            self.initialize_processor()

        self.tokenizer = self.processor.tokenizer

        self.label_pad_token_id = -100 if self.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        self.pad_to_multiple_of = 8 if self.fp16 else None

    def initialize_config(self):
        """

        """
        self.config = AutoConfig.from_pretrained(
            self.config_name if self.config_name else self.model_name_or_path,
            cache_dir=self.cache_dir,
            revision=self.model_revision,
            token=self.token,
            trust_remote_code=self.trust_remote_code,
        )

    def initialize_model(self):
        """

        """
        # if config is not given, try to initialize config
        if self.config is None:
            self.initialize_config()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_name_or_path),
            config=self.config,
            cache_dir=self.cache_dir,
            revision=self.model_revision,
            token=self.token,
            trust_remote_code=self.trust_remote_code,
        )

        logger.info(f"\n{self.model}\n")
        logger.info(f"\n{print_module_details(self.model)}\n")

        self.model.to(self.device)
        self.model.eval()

    def initialize_processor(self):
        """

        """
        self.processor = AutoProcessor.from_pretrained(self.processor_name_or_path, **self.processor_kwargs)
        for key in set(self.processor_kwargs.keys()):
            if hasattr(self.processor, key):
                setattr(self.processor, key, self.processor_kwargs.pop(key))


    def translate_tsv(self, tsv_path: str) -> TranslationOutput:
        """
        Translates a tab-separated values (TSV) file using the provided model and processor.

        Args:
            tsv_path: Path to the input TSV file to be translated.

        Returns:
            A TranslationOutput object containing the translation results, including the generated text and additional metadata.
        """

        logger.info(f"*** Translating input file {tsv_path} ***")

        return batched_inference(model=self.model,
                                 processor=self.processor,
                                 tsv_path=tsv_path,
                                 modality=self.modality,
                                 batch_size=self.batch_size,
                                 pad_to_multiple_of=self.pad_to_multiple_of,
                                 label_pad_token_id=self.label_pad_token_id,
                                 gen_kwargs=self.gen_kwargs,
                                 generation_config=self.generation_config)


def main():

    generate_args, extra_args, model_args, processor_args, data_args, training_args = assemble_generation_args()

    set_up_logging(log_level=training_args.get_process_log_level(),
                   should_log=training_args.should_log)

    # Telemetry (optional)
    send_example_telemetry("run_translate_only", model_args, data_args)

    # argument handling

    if generate_args.input is None:
        raise ValueError("Must specify an input TSV file with --input.")
    if generate_args.output is None:
        raise ValueError("Must specify an output JSON file for predictions with --output.")

    gen_kwargs = _infer_generation_kwargs(generate_args)

    # --- Processor / tokenizer ---
    if not processor_args.processor_name_or_path:
        raise ValueError("You must specify processor_name_or_path in the config or on the command line")

    processor_kwargs = construct_kwargs(processor_args, ["processor_name_or_path"])

    batch_size = getattr(training_args, "per_device_eval_batch_size", None) or 8
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    modality: ModalityType = "pose2text" # TODO: not hard-coded

    translator = Translator(config=model_args.config_name,
                            model_name_or_path=model_args.model_name_or_path,
                            processor_name_or_path=processor_args.processor_name_or_path,
                            processor_kwargs=processor_kwargs,
                            modality=modality,
                            batch_size=batch_size,
                            gen_kwargs=gen_kwargs,
                            seed=training_args.seed,
                            cache_dir=model_args.cache_dir,
                            model_revision=model_args.model_revision,
                            token=model_args.token,
                            trust_remote_code=model_args.trust_remote_code,
                            fp16=training_args.fp16,
                            ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
                            device=device)

    result = translator.translate_tsv(tsv_path=generate_args.input)

    # Save predictions as JSON

    logger.info(f"Saving translations to: {generate_args.output}")
    save_dict_to_json(data=result, path=generate_args.output)


if __name__ == "__main__":
    main()
