import argparse
import os
import shutil
import logging
import random
import time
import yaml

from peft.peft_model import PeftModel
from peft.tuners.ia3 import IA3Model
from peft.tuners.lora import LoraModel
from text_correction_utils.configuration import (
    load_config
)

from text_correction_utils.io import load_text_file

from deep_sparql.api.generator import SPARQLGenerator
from deep_sparql.model import QUANTIZATION_SCHEMES, PretrainedDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        required=True
    )
    parser.add_argument(
        "-s",
        "--scheme",
        choices=QUANTIZATION_SCHEMES,
        default=QUANTIZATION_SCHEMES[0]
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=16
    )
    parser.add_argument(
        "--use-triton",
        action="store_true"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None
    )
    return parser.parse_args()


def quantize(args: argparse.Namespace):
    gen = SPARQLGenerator.from_experiment(
        args.experiment,
        device="cpu"
    )
    assert isinstance(gen.model, PretrainedDecoder), \
        "only pretrained decoders can be quantized"

    inputs = load_text_file(args.input)
    targets = load_text_file(args.target)
    assert len(inputs) == len(targets), \
        "expected same number of inputs and targets"
    if args.max_examples is not None:
        # get random permutation
        perm = list(range(min(len(inputs), args.max_examples)))
        random.shuffle(perm)
        inputs = [inputs[i] for i in perm]
        targets = [targets[i] for i in perm]

    tok = gen.output_tokenizer
    examples = []
    for ipt, tgt in zip(inputs, targets):
        combined = f"{ipt}: {tgt}"
        token_ids = tok.tokenize(combined).token_ids
        examples.append({
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids)
        })

    info = load_config(os.path.join(args.experiment, "info.yaml"))
    cfg = load_config(os.path.join(args.experiment, info["config_name"]))
    cfg["model"] = {
        "type": "quantized_decoder",
        "path": "relpath(model)",
    }
    if isinstance(gen.model.model, PeftModel):
        if isinstance(gen.model.model.base_model, (LoraModel, IA3Model)):
            print("found lora/ia3 model, merging adapters into base model")
            gen.model.model = gen.model.model.base_model.merge_and_unload()
            cfg["train"].pop("peft", None)
        else:
            raise ValueError("unsupported peft type model")

    print(f"quantizing model\n{gen.model.model}\nwith scheme {args.scheme}")
    os.makedirs(args.output, exist_ok=True)
    # walk through experiment dir and copy all files to
    # output dir
    for path in os.listdir(args.experiment):
        full_path = os.path.join(args.experiment, path)
        if not os.path.isfile(full_path):
            continue
        shutil.copy2(full_path, os.path.join(args.output, path))

    with open(os.path.join(args.output, info["config_name"]), "w") as of:
        of.write(yaml.safe_dump(cfg))

    start = time.perf_counter()
    gen.model.quantize(
        args.scheme,
        os.path.join(args.output, "model"),
        examples,
        args.batch_size,
        args.use_triton
    )
    end = time.perf_counter()
    print(f"quantization took {end - start:.2f} seconds")


if __name__ == "__main__":
    quantize(parse_args())
