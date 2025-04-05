from typing import Dict, Any, Optional
from abc import abstractmethod

from .chat import AttributionTaskWithChatPrompt, DEFAULT_GENERATE_KWARGS


class ThoughtAttributionTask(AttributionTaskWithChatPrompt):
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        cache: Optional[Dict[str, Any]] = None,
        generate_kwargs: Dict[str, Any] = DEFAULT_GENERATE_KWARGS,
        **kwargs: Dict[str, Any],
    ):
        if not model.model_name_or_path.lower().startswith("deepseek-ai/deepseek-r1"):
            print(
                f"Warning: {model.model_name_or_path} is not a DeepSeek-R1 model."
                "This attribution task may not work as expected."
            )
        self._attribute_response = False
        self._target_token_start = None
        self._target_token_end = None
        super().__init__(
            model, tokenizer, cache=cache, generate_kwargs=generate_kwargs, **kwargs
        )

    def apply_chat_template(self, prompt):
        chat_prompt = super().apply_chat_template(prompt)
        chat_prompt += "<think>\n"
        return chat_prompt

    @property
    def thought_token_range(self):
        thought_end_marker = "</think>\n\n"
        tokens = self.get_tokens()
        if thought_end_marker in self.text:
            thought_end_marker_start = self.text.find(
                thought_end_marker, len(self.input_text)
            )
            thought_token_end = tokens.char_to_token(thought_end_marker_start)
        else:
            thought_token_end = self.generation_token_end
        return self.generation_token_start, thought_token_end

    @property
    def thoughts(self):
        thought_token_start, thought_token_end = self.thought_token_range
        tokens = self.get_tokens()
        start = tokens.token_to_chars(thought_token_start).start
        end = tokens.token_to_chars(thought_token_end - 1).end
        return self.text[start:end]

    @property
    def response_token_range(self):
        solution_start_marker = "</think>\n\n"
        if solution_start_marker in self.text:
            solution_start_marker_start = self.text.find(
                solution_start_marker, len(self.input_text)
            )
            solution_start = solution_start_marker_start + len(solution_start_marker)
            tokens = self.get_tokens()
            solution_token_start = tokens.char_to_token(solution_start)
            return solution_token_start, self.generation_token_end
        else:
            return self.generation_token_end, self.generation_token_end

    @property
    def response(self):
        response_token_start, response_token_end = self.response_token_range
        tokens = self.get_tokens()
        start = tokens.token_to_chars(response_token_start).start
        end = tokens.token_to_chars(response_token_end - 1).end
        return self.text[start:end]

    def _get_document_ranges(self):
        return []

    def _get_source_token_ranges(self):
        _, thought_token_end = self.thought_token_range
        if thought_token_end == self.generation_token_end:
            return []
        thought_token_ranges = self.get_sub_token_ranges(
            self.thought_token_range, split_by=self.source_type
        )
        source_token_ranges = super()._get_source_token_ranges()
        for source_token_start, source_token_end in thought_token_ranges:
            source_token_ranges.append((source_token_start, source_token_end))
        if self._attribute_response:
            target_token_ranges = self.get_sub_token_ranges(
                self.response_token_range,
                split_by=self.source_type,
            )
            for source_token_start, source_token_end in target_token_ranges:
                if source_token_end < self._target_token_start:
                    source_token_ranges.append((source_token_start, source_token_end))
        return source_token_ranges

    @property
    def target_token_range(self):
        if self._attribute_response:
            return self._target_token_start, self._target_token_end
        else:
            return self.response_token_range

    def attribute_response(self, token_start, token_end, relative=True):
        self._attribute_response = True
        if "source_token_ranges" in self._cache:
            del self._cache["source_token_ranges"]
        if "target_logits" in self._cache:
            del self._cache["target_logits"]
        if relative:
            response_token_start, _ = self.response_token_range
            self._target_token_start = response_token_start + token_start
            self._target_token_end = response_token_start + token_end
        else:
            self._target_token_start = token_start
            self._target_token_end = token_end


class SingleDocumentThoughtAttributionTask(ThoughtAttributionTask):
    template = "Context: {context}\n\nQuery: {query}"

    @property
    @abstractmethod
    def context(self) -> str:
        """The context."""

    @property
    @abstractmethod
    def query(self) -> str:
        """The query."""

    @property
    def prompt(self) -> str:
        return self.template.format(context=self.context, query=self.query)

    def _get_document_ranges(self):
        context_start = self.prompt.index(self.context)
        context_end = context_start + len(self.context)
        return [(context_start, context_end)]
