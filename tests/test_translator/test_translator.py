import pytest
import tempfile
import os
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

from multimodalhugs.tasks.translation.translate import Translator
from multimodalhugs.processors.text2text_preprocessor import Text2TextTranslationProcessor


class GenerateKwargsStrippingWrapper(nn.Module):
    """
    A wrapper around a model that strips certain model inputs from the generate call.
    """

    def __init__(self, base_model: nn.Module, model_input_ignore_keys: list = None):
        super().__init__()
        self.base_model = base_model  # register as submodule
        self.model_input_ignore_keys = model_input_ignore_keys if model_input_ignore_keys is not None else []

    def forward(self, *args, **kwargs):

        for ignore_key in self.model_input_ignore_keys:
            kwargs.pop(ignore_key, None)

        # Delegate forward pass
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):

        for ignore_key in self.model_input_ignore_keys:
            kwargs.pop(ignore_key, None)

        return self.base_model.generate(*args, **kwargs)

    def __getattr__(self, name):
        # Delegate everything else (config, device, etc.)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def test_translator_init_fails_without_args():
    with pytest.raises(ValueError):
        _ = Translator()


def test_translator_init_fails_without_processor_path():
    with pytest.raises(ValueError):
        _ = Translator(model_name_or_path="Helsinki-NLP/opus-mt-en-de")


@pytest.fixture
def create_temporary_tsv_file(request) -> str:

    identical_decoder_prompts = request.param["identical_decoder_prompts"]

    data = "signal\tdecoder_prompt\toutput\n"

    if identical_decoder_prompts:
        data += "hello my fellow engineers\tTranslate from English to German: \t\n"
        data += "this is a test\tTranslate from English to German: \t\n"
    else:
        data += "hello my fellow engineers\tTranslate from English to German: \t\n"
        data += "Das ist ein Test\tTranslate from German to English: \t\n"

    fd, path = tempfile.mkstemp(suffix=".tsv")
    os.write(fd, data.encode())
    os.close(fd)
    return path


@pytest.mark.parametrize(
    "create_temporary_tsv_file",
    [
        {
            "id": "identical_decoder_prompts",
            "identical_decoder_prompts": True,
        },
{
            "id": "different_decoder_prompts",
            "identical_decoder_prompts": False,
        },
    ],
    indirect=True,
)
def test_translator_text_to_text_generation(create_temporary_tsv_file):

    tsv_path = create_temporary_tsv_file

    base_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

    model_input_ignore_keys = ["encoder_prompt", "encoder_prompt_length_padding_mask"]
    model: nn.Module = GenerateKwargsStrippingWrapper(base_model,
                                                      model_input_ignore_keys)

    config = AutoConfig.from_pretrained("google-t5/t5-small")

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

    processor = Text2TextTranslationProcessor(tokenizer=tokenizer)

    translator = Translator(model=model,
                            config=config,
                            processor=processor,
                            modality="text2text")

    translator.translate_tsv(tsv_path)