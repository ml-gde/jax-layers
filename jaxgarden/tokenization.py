# type: ignore
"""Provides utility class for tokenization tasks such as encoding and decoding"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import jinja2
from huggingface_hub import hf_hub_download
from jinja2 import Environment, meta
from tokenizers import Tokenizer as HfTokenizer
from tokenizers.processors import TemplateProcessing

logger = logging.getLogger(__name__)

# Define conversation format
Message = dict[Literal["role", "content"], str]
Conversation = list[Message]


class Tokenizer:
    """
    Jax-friendly Tokenizer wrapper around Hugging Face's `tokenizers` library.

    Provides utilities for encoding text and applying chat templates, similar to
    `transformers.AutoTokenizer`, but returning JAX arrays.

    Attributes:
        hf_tokenizer: The underlying tokenizer instance from the `tokenizers` library.
        chat_template: The Jinja chat template string.
        bos_token: Beginning of sequence token string.
        eos_token: End of sequence token string.
        pad_token: Padding token string.
        bos_token_id: Beginning of sequence token ID.
        eos_token_id: End of sequence token ID.
        pad_token_id: Padding token ID.
    """

    def __init__(
        self,
        hf_tokenizer: HfTokenizer,
        chat_template: str | None = None,
        bos_token: str | None = None,
        eos_token: str | None = None,
        pad_token: str | None = None,
    ):
        """
        Initializes the Tokenizer.

        Args:
            hf_tokenizer: An initialized tokenizer from the `tokenizers` library.
            chat_template: Jinja template string for chat formatting.
            bos_token: BOS token string. If None, tries to get from tokenizer.
            eos_token: EOS token string. If None, tries to get from tokenizer.
            pad_token: Padding token string. If None, tries to get from tokenizer or uses EOS.
        """
        self.hf_tokenizer = hf_tokenizer
        self.chat_template = chat_template

        # --- Special Tokens ---
        def _resolve_token(token, hf_tokenizer, fallback_names):
            if token is not None:
                if isinstance(token, int):
                    return hf_tokenizer.id_to_token(token)
                return token
            # fallbacks
            for name in fallback_names:
                tid = hf_tokenizer.token_to_id(name)
                if tid is not None:
                    return name
            return None

        self.bos_token = _resolve_token(bos_token, self.hf_tokenizer, ["[BOS]", "<s>"])
        self.eos_token = _resolve_token(eos_token, self.hf_tokenizer, ["[EOS]", "</s>"])

        # Padding token handling
        if pad_token is not None:
            if isinstance(pad_token, int):
                self.pad_token = self.hf_tokenizer.id_to_token(pad_token)
            else:
                self.pad_token = pad_token
        elif self.hf_tokenizer.padding and self.hf_tokenizer.padding.get("pad_token"):
            self.pad_token = self.hf_tokenizer.padding["pad_token"]
        elif self.eos_token:
            logger.warning("Using EOS token as pad token.")
            self.pad_token = self.eos_token
        else:
            pad_tok_id = self.hf_tokenizer.token_to_id("[PAD]") or self.hf_tokenizer.token_to_id(
                "<pad>"
            )
            if pad_tok_id is not None:
                self.pad_token = self.hf_tokenizer.id_to_token(pad_tok_id)
                logger.warning(f"Using found token {self.pad_token} as pad token.")
            else:
                raise ValueError(
                    "Cannot determine padding token. Please provide `pad_token` explicitly."
                )

        # --- Special Token IDs ---
        self.eos_token_id = (
            self.hf_tokenizer.token_to_id(self.eos_token) if self.eos_token else None
        )
        self.bos_token_id = (
            self.hf_tokenizer.token_to_id(self.bos_token) if self.bos_token else None
        )
        self.pad_token_id = self.hf_tokenizer.token_to_id(self.pad_token)

        if self.pad_token_id is None:
            raise ValueError(f"Padding token '{self.pad_token}' not found in tokenizer vocabulary.")

        # --- Configure Padding ---
        if self.hf_tokenizer.padding is None:
            logger.info(
                "Configuring tokenizer padding with"
                + f" pad_id={self.pad_token_id}, pad_token='{self.pad_token}'"
            )
            self.hf_tokenizer.enable_padding(pad_id=self.pad_token_id, pad_token=self.pad_token)
        else:
            logger.info("Tokenizer already has padding configured. Verifying PAD token ID.")
            if self.hf_tokenizer.padding["pad_id"] != self.pad_token_id:
                logger.warning(
                    f"Tokenizer PAD ID ({self.hf_tokenizer.padding['pad_id']}) differs from"
                    + f" derived PAD ID ({self.pad_token_id}). Using derived ID."
                )
                self.hf_tokenizer.enable_padding(
                    pad_id=self.pad_token_id,
                    pad_token=self.pad_token,
                    length=self.hf_tokenizer.padding.get("length"),
                )  # Keep length if already set

    @classmethod
    def from_pretrained(cls, identifier: str, **kwargs) -> "Tokenizer":
        """
        Loads a tokenizer from the Hugging Face Hub or a local path.

        Args:
            identifier: The repository ID on the Hub (e.g., "gpt2") or path to a
                        directory containing tokenizer files ('tokenizer.json').
            **kwargs: Additional keyword arguments passed to the constructor
                      (e.g., chat_template).

        Returns:
            An instance of the Tokenizer.
        """
        is_local = Path(identifier).exists()
        if is_local:
            tokenizer_path = Path(identifier) / "tokenizer.json"
            config_path = Path(identifier) / "tokenizer_config.json"
            if not tokenizer_path.is_file():
                raise FileNotFoundError(f"tokenizer.json not found in {identifier}")
            hf_tok = HfTokenizer.from_file(str(tokenizer_path))
            local_config = {}
            if config_path.is_file():
                with open(config_path) as f:
                    local_config = json.load(f)
        else:
            # Download tokenizer.json
            try:
                tokenizer_path_hub = hf_hub_download(repo_id=identifier, filename="tokenizer.json")
                hf_tok = HfTokenizer.from_file(tokenizer_path_hub)
            except Exception as e:
                logger.error(f"Failed to download or load tokenizer.json for {identifier}: {e}")
                raise

            # Download tokenizer_config.json for chat template and special tokens
            local_config = {}
            try:
                config_path_hub = hf_hub_download(
                    repo_id=identifier, filename="tokenizer_config.json"
                )
                with open(config_path_hub) as f:
                    local_config = json.load(f)
            except Exception:
                logger.warning(
                    f"tokenizer_config.json not found for {identifier}."
                    + " Chat template and special tokens might be missing."
                )

        # Extract relevant info from config for constructor
        chat_template = kwargs.pop("chat_template", local_config.get("chat_template"))
        bos_token = kwargs.pop("bos_token", local_config.get("bos_token"))
        eos_token = kwargs.pop("eos_token", local_config.get("eos_token"))
        pad_token = kwargs.pop("pad_token", local_config.get("pad_token"))

        return cls(
            hf_tok,
            chat_template=chat_template,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    @staticmethod
    def _validate_template(template: str, variables: set) -> None:
        """Checks if template variables match expected ones."""
        env = Environment()
        try:
            parsed_content = env.parse(template)
            template_vars = meta.find_undeclared_variables(parsed_content)
            required_vars = {"messages", "bos_token", "eos_token"}  # Common variables
            missing_vars = required_vars - template_vars
            extra_vars = template_vars - required_vars - variables  # Allow extra user-defined vars

            if missing_vars:
                logger.warning(f"Chat template might be missing expected variables: {missing_vars}")
            if extra_vars:
                logger.warning(f"Chat template uses potentially unexpected variables: {extra_vars}")

        except Exception as e:
            logger.error(f"Failed to parse chat template: {e}")
            raise ValueError(f"Invalid Jinja template: {template}") from e

    def apply_chat_template(
        self,
        conversation: Conversation,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> str | list[int]:
        """
        Formats a conversation using the tokenizer's chat template.

        Args:
            conversation: A list of dictionaries, where each dictionary has keys
                          'role' and 'content'. Example:
                          [{'role': 'user', 'content': 'Hello'},
                          {'role': 'assistant', 'content': 'Hi!'}]
            chat_template: An optional Jinja template string to override the default.
            add_generation_prompt: If True, adds the template's prompt for the assistant's
                                   next reply.
            **kwargs: Additional variables to pass to the Jinja template.

        Returns:
            The formatted string.
        Raises:
            ValueError: If no chat template is available or Jinja2 is not installed.
        """
        template = chat_template or self.chat_template
        if template is None:
            raise ValueError(
                "No chat template defined for this tokenizer."
                + " Provide `chat_template` argument or initialize with one."
            )

        # Prepare template variables
        template_vars = {
            "messages": conversation,
            "bos_token": self.bos_token or "",
            "eos_token": self.eos_token or "",
            # You might need to add other special tokens depending on common templates
            **kwargs,
        }

        # Basic validation (can be expanded)
        self._validate_template(template, set(kwargs.keys()))

        # Add generation prompt if requested (common pattern in HF templates)
        if add_generation_prompt:
            template += "{{ bos_token if messages[-1]['role'] == 'assistant' else '' }}{% if add_generation_prompt %}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'user:' }}{% elif message['role'] == 'assistant' %}{{ 'assistant:' }}{% endif %}{{ message['content'] }}{% endfor %}{{ 'assistant:'}}{% endif %}"  # noqa: E501

        env = jinja2.Environment().from_string(template)

        try:
            rendered = env.render(**template_vars)
            return rendered
        except Exception as e:
            logger.error(f"Error rendering chat template: {e}")
            raise ValueError("Failed to render chat template. Check template and variables.") from e

    def encode(
        self,
        text: str | list[str],
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: Literal["jax"] | None = "jax",
    ) -> dict[str, jnp.ndarray] | dict[str, list[int]]:
        """
        Encodes text or a batch of text into token IDs and attention masks.

        Args:
            text: A single string or a list of strings to encode.
            add_special_tokens: Whether to add BOS/EOS tokens.
            padding: If True or 'longest', pads to the longest sequence in the batch.
                     If 'max_length', pads to `max_length`.
                     If False, no padding.
            truncation: Whether to truncate sequences to `max_length`. Requires `max_length`.
            max_length: Maximum sequence length for padding or truncation.
            return_tensors: If "jax", returns JAX arrays. Otherwise, returns lists of ints.

        Returns:
            A dictionary containing 'input_ids' and 'attention_mask'.
            Values are JAX arrays if return_tensors='jax', otherwise lists of integers.
        """
        is_batch = isinstance(text, list)

        # Configure truncation
        if truncation:
            if max_length is None:
                raise ValueError("`truncation=True` requires `max_length` to be set.")
            self.hf_tokenizer.enable_truncation(max_length=max_length)
        else:
            self.hf_tokenizer.no_truncation()

        # Configure padding
        if padding:
            pad_strategy = "longest" if padding is True else padding
            if pad_strategy == "longest":
                self.hf_tokenizer.enable_padding(
                    pad_id=self.pad_token_id, pad_token=self.pad_token, direction="right"
                )
            elif pad_strategy == "max_length":
                if max_length is None:
                    raise ValueError("padding='max_length' requires `max_length` to be set.")
                self.hf_tokenizer.enable_padding(
                    pad_id=self.pad_token_id,
                    pad_token=self.pad_token,
                    length=max_length,
                    direction="right",
                )
            else:
                raise ValueError(
                    f"Invalid padding strategy: {padding}. Choose True, 'longest', or 'max_length'."
                )
        else:
            self.hf_tokenizer.no_padding()

        # Add BOS/EOS if requested (using TemplateProcessing)
        # Note: This assumes a simple structure. More complex templates might need manual handling.
        if add_special_tokens and self.bos_token and self.eos_token:
            self.hf_tokenizer.post_processor = TemplateProcessing(
                single=f"{self.bos_token} $A {self.eos_token}",
                pair=f"{self.bos_token} $A {self.eos_token} {self.bos_token} $B {self.eos_token}",
                special_tokens=[
                    (self.bos_token, self.bos_token_id),
                    (self.eos_token, self.eos_token_id),
                ],
            )
        else:
            # Disable post-processing if not adding special tokens or if tokens are missing
            self.hf_tokenizer.post_processor = None

        # Encode
        if is_batch:
            encoded_list = self.hf_tokenizer.encode_batch(text)
        else:
            encoded_list = [self.hf_tokenizer.encode(text)]

        # Prepare output
        input_ids = [enc.ids for enc in encoded_list]
        attention_mask = [enc.attention_mask for enc in encoded_list]

        if return_tensors == "jax":
            # Find max length *after* encoding if padding strategy was 'longest' and it's a batch
            if padding == "longest" or (padding is True and is_batch):
                actual_max_len = max(len(ids) for ids in input_ids)
                # Re-pad manually to ensure rectangular JAX array if needed
                # (HF tokenizer might already do this)
                input_ids = [
                    ids + [self.pad_token_id] * (actual_max_len - len(ids)) for ids in input_ids
                ]
                attention_mask = [
                    mask + [0] * (actual_max_len - len(mask)) for mask in attention_mask
                ]

            return {
                "input_ids": jnp.array(input_ids, dtype=jnp.int32),
                "attention_mask": jnp.array(attention_mask, dtype=jnp.int32),
            }
        else:
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(
        self,
        token_ids: list[int] | list[list[int]] | jnp.ndarray,
        skip_special_tokens: bool = True,
    ) -> str | list[str]:
        """
        Decodes token IDs back into strings.

        Args:
            token_ids: A single list of token IDs, a batch (list of lists), or a JAX array.
            skip_special_tokens: Whether to remove special tokens (BOS, EOS, PAD) from the output.

        Returns:
            A single decoded string or a list of decoded strings.
        """
        if isinstance(token_ids, jnp.ndarray):
            token_ids = token_ids.tolist()  # Convert JAX array to list

        if not token_ids:
            return (
                ""
                if not isinstance(token_ids, list)
                or not (token_ids and isinstance(token_ids[0], list))
                else []
            )

        is_batch = isinstance(token_ids[0], list)

        if is_batch:
            return self.hf_tokenizer.decode_batch(
                token_ids, skip_special_tokens=skip_special_tokens
            )
        else:
            # Ensure it's a list of integers for single decode
            if not all(isinstance(x, int) for x in token_ids):
                raise TypeError("Input for single sequence decoding must be a list of integers.")
            return self.hf_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        """Returns the size of the tokenizer vocabulary."""
        return self.hf_tokenizer.get_vocab_size(with_added_tokens=True)
