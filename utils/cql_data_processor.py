"""Custom NeMo RL data processor for CQL RLVR.

Handles system/user/assistant separation with proper chat template application.
Works for both GRPO (generate assistant) and SFT (learn from golden assistant).

NeMo RL's default processor (math_hf_data_processor) ignores system_prompt_file.
This processor correctly creates a multi-role message log:
  - system: CQL expert prompt with few-shot examples (from system_prompt_file)
  - user:   "Schema: ...\nRequest: <nl_query>"  (the input_key field)
  - assistant (SFT only): golden CQL query for supervised learning

For GRPO: system + user only (model generates assistant at inference time).
For SFT:  system + user + assistant (model learns to reproduce golden CQL).

Registration:
    # In your launch script or training entrypoint, before NeMo RL starts:
    from utils.cql_data_processor import register_cql_processor
    register_cql_processor()

    # Then in your NeMo RL config:
    # data.default.processor: "cql_data_processor"
"""

import logging
from typing import Any, Optional, Union, cast

import torch
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases matching NeMo RL's nemo_rl.data.interfaces
# ---------------------------------------------------------------------------
LLMMessageLogType = list[dict[str, Union[str, torch.Tensor]]]
TokenizerType = PreTrainedTokenizerBase


def cql_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: Any,  # nemo_rl.data.interfaces.TaskDataSpec
    tokenizer: TokenizerType,
    max_seq_length: Optional[int],
    idx: int,
    add_bos: bool = False,
    add_eos: bool = False,
    add_generation_prompt: bool = True,
) -> dict[str, Any]:
    """Process a single CQL training example into a NeMo RL DatumSpec.

    Expected datum_dict keys (set by ResponseDataset from JSONL):
        messages[0]["content"] → the user's NL query (input_key value)
        messages[1]["content"] → the golden CQL query (output_key value)

    Modes:
        add_generation_prompt=True  (GRPO): system + user only, model generates
        add_generation_prompt=False (SFT):  system + user + assistant (golden CQL)

    The processor:
        1. Extracts the NL question and golden CQL from messages
        2. Optionally prepends a system prompt (from system_prompt_file)
        3. Tokenizes each role using the model's chat template
        4. For SFT: includes golden CQL as assistant message for training
        5. For GRPO: stores golden CQL in extra_env_info for reward computation
    """
    messages = datum_dict.get("messages")
    if not messages or len(messages) < 2:
        raise ValueError(
            f"datum_dict at idx={idx} missing 'messages' with ≥2 entries. "
            f"Keys present: {list(datum_dict.keys())}"
        )
    user_question = messages[0]["content"]       # nl_query from JSONL
    golden_cql = messages[1]["content"]           # cql_query from JSONL

    extra_env_info = {"ground_truth": golden_cql}
    message_log: LLMMessageLogType = []

    # ── System prompt (from system_prompt_file → task_data_spec.system_prompt) ──
    if task_data_spec.system_prompt:
        sys_msg: dict[str, Any] = {
            "role": "system",
            "content": task_data_spec.system_prompt,
        }
        sys_text = tokenizer.apply_chat_template(
            [cast(dict[str, str], sys_msg)],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        sys_ids = tokenizer(
            sys_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        # Prepend BOS to first message if requested
        if add_bos and hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            sys_ids = torch.cat([torch.tensor([tokenizer.bos_token_id]), sys_ids])
        sys_msg["token_ids"] = sys_ids
        message_log.append(sys_msg)

    # ── User message ──
    if task_data_spec.prompt:
        user_content = task_data_spec.prompt.format(user_question)
    else:
        user_content = user_question

    user_msg: dict[str, Any] = {"role": "user", "content": user_content}
    user_text = tokenizer.apply_chat_template(
        [user_msg],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        add_special_tokens=False,
    )
    user_ids = tokenizer(
        user_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    # If no system message, BOS goes on user message
    if add_bos and not message_log and hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
        user_ids = torch.cat([torch.tensor([tokenizer.bos_token_id]), user_ids])
    user_msg["token_ids"] = user_ids
    user_msg["content"] = user_text
    message_log.append(user_msg)

    # ── Assistant message (SFT mode only) ──
    if not add_generation_prompt:
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": golden_cql,
        }
        assistant_text = tokenizer.apply_chat_template(
            [assistant_msg],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        assistant_ids = tokenizer(
            assistant_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        # Append EOS to last message if requested
        if add_eos and hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            assistant_ids = torch.cat([assistant_ids, torch.tensor([tokenizer.eos_token_id])])
        assistant_msg["token_ids"] = assistant_ids
        message_log.append(assistant_msg)
    elif add_eos:
        # GRPO mode with add_eos — append EOS to user message (last message)
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            user_msg["token_ids"] = torch.cat([user_msg["token_ids"], torch.tensor([tokenizer.eos_token_id])])

    # ── Length and overflow handling ──
    length = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier = 1.0

    if max_seq_length and length > max_seq_length:
        logger.warning(
            "Sequence at idx=%d exceeds max_seq_length (%d > %d), truncating. "
            "loss_multiplier set to 0.0.", idx, length, max_seq_length
        )
        for msg in message_log:
            msg["token_ids"] = msg["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]

    return output


def register_cql_processor() -> None:
    """Register cql_data_processor in NeMo RL's processor registry.

    Call this before NeMo RL initializes datasets. For example, in
    your launch script or at the top of your training entrypoint.
    """
    try:
        from nemo_rl.data.processors import register_processor
        register_processor("cql_data_processor", cql_data_processor)
    except ImportError:
        # NeMo RL not installed — running in dummy/validation mode
        pass
