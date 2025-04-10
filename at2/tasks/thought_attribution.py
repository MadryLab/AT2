from typing import Dict, Any, Optional, List

from .chat import AttributionTaskWithChatPrompt


# Higher `max_new_tokens` to accommodate the intermediate thoughts
DEFAULT_GENERATE_KWARGS = {
    "max_new_tokens": 4096,
    "do_sample": False,
    "top_p": None,
    "top_k": None,
    "temperature": None,
}


class ThoughtAttributionTask(AttributionTaskWithChatPrompt):
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        cache: Optional[Dict[str, Any]] = None,
        generate_kwargs: Dict[str, Any] = DEFAULT_GENERATE_KWARGS,
        **kwargs: Dict[str, Any],
    ):
        if not model.name_or_path.lower().startswith("deepseek-ai/deepseek-r1"):
            # We use </think> to identify the end of the thought
            print(
                f"Warning: {model.name_or_path} is not a DeepSeek-R1 model."
                "This attribution task may not work as expected."
            )
        self._target_token_start = None
        self._target_token_end = None
        super().__init__(
            model, tokenizer, cache=cache, generate_kwargs=generate_kwargs, **kwargs
        )

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
            raise ValueError("No response start marker found in the model's generation.")

    @property
    def response(self):
        response_token_start, response_token_end = self.response_token_range
        tokens = self.get_tokens()
        start = tokens.token_to_chars(response_token_start).start
        end = tokens.token_to_chars(response_token_end - 1).end
        return self.text[start:end]

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
        return source_token_ranges

    @property
    def target_token_range(self):
        return self.response_token_range


class SimpleThoughtAttributionTask(ThoughtAttributionTask):
    def __init__(
        self,
        query: str,
        model: Any,
        tokenizer: Any,
        generate_kwargs: Dict[str, Any] = DEFAULT_GENERATE_KWARGS,
        previous_messages: Optional[List[Dict[str, str]]] = None,
        source_type: str = "token",
        cache: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Any],
    ):
        self.query = query
        super().__init__(
            model,
            tokenizer,
            cache=cache,
            generate_kwargs=generate_kwargs,
            previous_messages=previous_messages,
            source_type=source_type,
            **kwargs,
        )

    @property
    def prompt(self) -> str:
        return self.query

    def _get_document_ranges(self):
        return []
