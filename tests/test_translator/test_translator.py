import pytest
import tempfile
import os

from transformers import AutoProcessor

from multimodalhugs.tasks.translation.translate import Translator
from multimodalhugs.processors.text2text_preprocessor import Text2TextTranslationProcessor


def test_translator_init_fails_without_args():
    with pytest.raises(ValueError):
        _ = Translator()


def test_translator_init_fails_without_processor_path():
    with pytest.raises(ValueError):
        _ = Translator(model_name_or_path="Helsinki-NLP/opus-mt-en-de")

@pytest.fixture
def create_temporary_tsv_file() -> str:

    data = "signal\tencoder_prompt\tdecoder_prompt\toutput\n"
    data += "hello my fellow engineers\t\tTranslate from English to German: \t\n"
    data += "this is a test\t\tTranslate from English to German: \t\n"

    fd, path = tempfile.mkstemp(suffix=".tsv")
    os.write(fd, data.encode())
    os.close(fd)
    return path


def test_translator_text_to_text_generation(create_temporary_tsv_file):

    tsv_path = create_temporary_tsv_file

    tokenizer = AutoProcessor.from_pretrained("google-t5/t5-small")

    processor = Text2TextTranslationProcessor(tokenizer=tokenizer)

    translator = Translator(model_name_or_path="google-t5/t5-small",
                            processor=processor,
                            modality="text2text")

    translator.translate_tsv(tsv_path)