# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from time import perf_counter

import numpy as np
import requests
import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

model_id = "Qwen/Qwen3-VL-235B-A22B-Instruct"
# model_id = "tiny-random/qwen3-vl-moe"
config = AutoConfig.from_pretrained(model_id)
config.dtype = "float16"
config.torch_dtype = torch.float16

# For faster execution user can run with lesser layers, For Testing Purpose Only
config.vision_config.depth = 9
config.text_config.num_hidden_layers = 2
config.vision_config.deepstack_visual_indexes = [8]

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id, attn_implementation="eager", kv_offload=True, config=config, dtype=torch.float16, layerwise=False
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

PREFILL_SEQ_LEN = 1024
CTX_LEN = 2048
BS = 1

NUM_KV_BLOCKS = 4
NUM_Q_BLOCKS = 2
HEAD_BLOCK_SIZE = 8
PREFILL_BLOCK_CHUNKS = None
PREFILL_MODE = None  # None, "online" or "qkv" depending on whether we want online prefill or headparallel prefill


###############
# Decode modes:
# - standard attention - pass enable_blocking and blocking_mode
# - head parallel blocking - pass  enable_blocking, blocking_mode: “kv” and kv_block_headpar_split: 0
# - batch fold head parallel - pass enable_blocking, blocking_mode: “kv” and batch_fold: True


def _decode_qaic_config() -> dict:
    return {
        "blocking_mode": "kv",
        "num_kv_blocks": NUM_KV_BLOCKS,
        # "kv_blocking_headpar_split": 0,  # 0 → resolved to num_cores at compile time
        "batch_fold": True,
        "ctx_len": CTX_LEN,
    }


def _qaic_config() -> dict:
    cfg = _decode_qaic_config()
    if PREFILL_MODE is None:
        return cfg
    cfg["prefill_block_chunks"] = PREFILL_BLOCK_CHUNKS
    cfg["prefill_blocking_mode"] = PREFILL_MODE
    cfg["prefill_n_rep_chunk"] = PREFILL_N_REP_CHUNK
    return cfg


skip_vision = True
if not skip_vision:
    vision_qpc_path = qeff_model.compile(
        batch_size=BS,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        height=354,
        width=536,
        num_cores=16,
        num_devices=1,
        mos=1,
        mxfp6_matmul=True,
        aic_enable_depth_first=True,
        skip_vision=skip_vision,
        split_model_io=True,
        skip_lang=True,
        use_onnx_subfunctions=True,
        layerwise=False,
    )
decode_qaic_config = _qaic_config()
print("decode", decode_qaic_config)
decode_qpc_path = qeff_model.compile(
    batch_size=BS,
    prefill_seq_len=1,
    ctx_len=CTX_LEN,
    height=354,
    width=536,
    num_cores=16,
    num_devices=1,
    mxfp6_matmul=True,
    split_model_io=True,
    mos=1,
    user_tiled=True,
    prefill_only=False,
    skip_vision=True,
    use_onnx_subfunctions=False,
    layerwise=False,
    offload_pt_weights=False,
    qaic_config=decode_qaic_config,
)

################
# Prefill modes:
# - follow decode attention - pass nothing extra
# - head parallel offline prefill - pass prefill_blocking_mode: “qkv”, prefill_block_chunks: 2
# - online prefill - pass prefill_blocking_mode: “online”, prefill_block_chunks: 2
PREFILL_MODE = "online"
PREFILL_QL_CHUNK = 128
PREFILL_BLOCK_CHUNKS = -(-PREFILL_SEQ_LEN // PREFILL_QL_CHUNK)
PREFILL_N_REP_CHUNK = 4
MOE_PREFILL_PACKED_CHUNK_SIZE = 256
prefill_qaic_config = _qaic_config()
print("prefill", prefill_qaic_config)

# Skip prefill compile — decode-only run for trace collection
prefill_qpc_path = None

print(f"Decode qpc path {decode_qpc_path}")

lang_decode_session = QAICInferenceSession(decode_qpc_path.get("lang_decode_qpc_path"))

# Synthetic decode run — no prefill needed, just measure decode latency
num_layers = config.text_config.num_hidden_layers
num_kv_heads = config.text_config.num_key_value_heads
head_dim = config.text_config.hidden_size // config.text_config.num_attention_heads

decode_inputs = {
    "input_ids":    np.zeros((BS, 1), dtype=np.int32),
    "position_ids": np.zeros((BS, 1), dtype=np.int32),
}
for i in range(num_layers):
    decode_inputs[f"past_key.{i}"]   = np.zeros((1, num_kv_heads, CTX_LEN, head_dim), dtype=np.float16)
    decode_inputs[f"past_value.{i}"] = np.zeros((1, num_kv_heads, CTX_LEN, head_dim), dtype=np.float16)

# warmup
lang_decode_session.run(decode_inputs)

st = perf_counter()
decode_out = lang_decode_session.run(decode_inputs)
print(f"Decode latency (first measured) = {perf_counter() - st:.4f} sec")
