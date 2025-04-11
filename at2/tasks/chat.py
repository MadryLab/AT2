from abc import abstractmethod
from typing import Dict, Any, Optional, List

from .base import AttributionTask, DEFAULT_GENERATE_KWARGS
from ..utils import split_text


class AttributionTaskWithChatPrompt(AttributionTask):
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        generate_kwargs: Dict[str, Any] = DEFAULT_GENERATE_KWARGS,
        previous_messages: Optional[List[Dict[str, str]]] = None,
        source_type: str = "document",
        cache: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(
            model, tokenizer, cache=cache, generate_kwargs=generate_kwargs, **kwargs
        )
        self.previous_messages = [] if previous_messages is None else previous_messages
        self.source_type = source_type
        self._source_to_document = []

    @property
    @abstractmethod
    def prompt(self):
        """The prompt."""

    def apply_chat_template(self, prompt):
        messages = self.previous_messages.copy()
        messages += [{"role": "user", "content": prompt}]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return chat_prompt

    @property
    def input_text(self):
        return self.apply_chat_template(self.prompt)

    def prompt_range_to_token_range(self, start_index, end_index):
        _, chat_prompt_tokens = self.get_input_text_and_tokens()
        # Find offset for chat template
        placeholder = "<placeholder>"
        placeholder_chat_prompt = self.apply_chat_template(placeholder)
        chat_offset_index = placeholder_chat_prompt.find(placeholder)
        token_start_index = chat_prompt_tokens.char_to_token(
            start_index + chat_offset_index
        )
        token_end_index = (
            chat_prompt_tokens.char_to_token(end_index + chat_offset_index - 1) + 1
        )
        return token_start_index, token_end_index

    @property
    def document_ranges(self):
        if self._cache.get("document_ranges") is None:
            self._cache["document_ranges"] = self._get_document_ranges()
        return self._cache["document_ranges"]

    @abstractmethod
    def _get_document_ranges(self):
        """Get the ranges of the prompt corresponding to documents. If source_type is "document", the documents are the sources. Otherwise, each document is split into individual sources."""

    def _get_source_token_ranges(self):
        source_token_ranges = []
        self._source_to_document = []
        for document_index, (document_start, document_end) in enumerate(
            self.document_ranges
        ):
            if self.source_type == "document":
                token_start, token_end = self.prompt_range_to_token_range(
                    document_start, document_end
                )
                source_token_ranges.append((token_start, token_end))
                self._source_to_document.append(document_index)
            elif self.source_type == "token":
                token_start, token_end = self.prompt_range_to_token_range(
                    document_start, document_end
                )
                for i in range(token_start, token_end):
                    source_token_ranges.append((i, i + 1))
                    self._source_to_document.append(document_index)
            elif self.source_type == "word" or self.source_type == "sentence":
                document = self.prompt[document_start:document_end]
                _, _, indices = split_text(document, self.source_type)
                for start, end in indices:
                    token_start, token_end = self.prompt_range_to_token_range(
                        start + document_start, end + document_start
                    )
                    source_token_ranges.append((token_start, token_end))
                    self._source_to_document.append(document_index)
            else:
                raise ValueError(f"Invalid source_type: {self.source_type}")
        return source_token_ranges
