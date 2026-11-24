"""The chat templates RQ3's turns are written in (#209).

Each family's pieces against HuggingFace's own rendering of the same
conversation, which tools/gen_golden.py records beside the reference; and
the pieces Qwen2.5's tasks were built from before the templates were kept
here, which the logged runs' samples must still be, byte for byte.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import require_model
from microinfer import chat, needles
from microinfer.config import ModelConfig
from test_config import LLAMA

REPO = Path(__file__).resolve().parent.parent


def test_qwen_s_pieces_are_the_ones_its_samples_were_built_from():
    """needles' and rq3's constants, as #146 to #208 built every logged
    sample: the template moved here, its text did not."""
    t = chat.QWEN
    assert t.opening == ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a "
                         "helpful assistant.<|im_end|>\n<|im_start|>user\n")
    assert needles.TURN_1_OPENING == t.opening
    assert needles.TURN_1_CLOSING == ("\n\nSummarize the document above in two sentences."
                                      "<|im_end|>\n<|im_start|>assistant\n")
    assert needles.TURN_2_AFTER_EOS == "\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    assert needles.END_OF_TURN == "<|im_end|>"


def test_each_family_has_its_template():
    assert chat.template_for(ModelConfig.from_card("qwen2.5-0.5b-instruct")) is chat.QWEN
    assert chat.template_for(ModelConfig.from_dict("llama-3.2-1b-instruct", LLAMA)) is chat.LLAMA3


@pytest.mark.parametrize("name", ["qwen2.5-0.5b-instruct", "llama-3.2-1b-instruct"])
def test_the_template_renders_as_huggingface_does(name):
    """The probe conversation, opened for the assistant's reply, as
    HuggingFace's chat template rendered it for the reference (gen_golden);
    skipped for a reference made before it recorded one."""
    manifest = REPO / "tests" / "golden" / name / "manifest.json"
    if not manifest.is_file():
        pytest.skip(f"no reference for {name}; tools/gen_golden.py makes one")
    probe = json.loads(manifest.read_text()).get("chat_template_probe")
    if probe is None:
        pytest.skip(f"{name}'s reference predates the template probe; regenerate it")
    cfg = ModelConfig.from_model_dir(REPO / "models" / name)
    assert chat.template_for(cfg).render(*chat.PROBE) == probe


def test_a_piece_is_encoded_without_special_tokens():
    """Qwen2.5's tokenizer adds none, so a piece's ids are its plain
    encoding's: the logged samples' tokens stay as they were."""
    engine_tokenizer = _tokenizer("qwen2.5-0.5b-instruct")
    text = " The access code for the Juniper archive is XQT-4827.\n"
    np.testing.assert_array_equal(chat.encode(engine_tokenizer, text),
                                  engine_tokenizer.encode(text).ids)
    batch = chat.encode_batch(engine_tokenizer, [text, "<|im_end|>"])
    assert [b.tolist() for b in batch] == [engine_tokenizer.encode(t).ids
                                           for t in (text, "<|im_end|>")]


def test_llama_s_pieces_carry_no_begin_of_text_but_its_opening_does():
    """Llama-3.2's tokenizer adds <|begin_of_text|> to an encoding; a piece
    tokenized apart must not carry one, and the opening carries exactly one,
    its first token. Its end of turn is one token."""
    tokenizer = _tokenizer("llama-3.2-1b-instruct")
    bos = tokenizer.token_to_id("<|begin_of_text|>")
    assert tokenizer.encode("Hello").ids[0] == bos
    assert bos not in chat.encode(tokenizer, " Hello world.\n").tolist()
    opening = chat.encode(tokenizer, chat.LLAMA3.opening).tolist()
    assert opening[0] == bos and opening.count(bos) == 1
    assert chat.encode(tokenizer, chat.LLAMA3.end_of_turn).tolist() == [
        tokenizer.token_to_id("<|eot_id|>")]


def _tokenizer(name: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(require_model(name) / "tokenizer.json"))


def test_the_generator_renders_the_same_probe_on_the_same_date():
    """gen_golden imports torch and cannot be imported here (ADR-0002), so
    its probe and date are read from its source: they must be chat's, or
    the comparison above compares different conversations."""
    import ast
    tree = ast.parse((REPO / "tools" / "gen_golden.py").read_text())
    found = {t.id: ast.literal_eval(node.value) for node in ast.walk(tree)
             if isinstance(node, ast.Assign) for t in node.targets
             if getattr(t, "id", None) in ("CHAT_PROBE", "CHAT_DATE")}
    assert found == {"CHAT_PROBE": chat.PROBE, "CHAT_DATE": chat.LLAMA3_DATE}


def test_llama_s_template_trims_a_message_s_content_and_qwen_s_does_not():
    """Llama-3.2's Jinja renders `message['content'] | trim`; Qwen2.5's
    renders the content as given. So the first piece of a Llama user
    message loses its leading space, as HuggingFace would render it, and a
    Qwen one keeps it: the logged Qwen samples are unchanged."""
    text = " The access code for the Juniper archive is XQT-4827.\n"
    llama = _tokenizer("llama-3.2-1b-instruct")
    ids = chat.encode(llama, text)
    np.testing.assert_array_equal(chat.content_opening(llama, chat.LLAMA3, ids),
                                  chat.encode(llama, text.lstrip()))
    qwen = _tokenizer("qwen2.5-0.5b-instruct")
    ids = chat.encode(qwen, text)
    assert chat.content_opening(qwen, chat.QWEN, ids) is ids
    assert chat.LLAMA3.trims and not chat.QWEN.trims


def test_the_needle_set_s_llama_turns_open_their_content_as_huggingface_would():
    """Turn 1's content, after the opening, starts with no space, at every
    depth, the 0% needle included; the needle's positions are still where
    its tokens are."""
    from conftest import load_tool
    tokenizer = _tokenizer("llama-3.2-1b-instruct")
    text = load_tool("perplexity").held_out_text()
    samples = needles.needle_set(text, tokenizer, seed=146, samples_per_depth=1,
                                 template=chat.LLAMA3)
    for s in samples:
        content = tokenizer.decode(s.turn_1[s.document_start:].tolist())
        assert not content[0].isspace(), s.sample_id
        start, end = s.needle_positions
        assert s.needle.lstrip(" ") in tokenizer.decode(s.turn_1[start:end].tolist())
