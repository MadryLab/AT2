from typing import Tuple, Any, Dict, Optional, List

from .chat import AttributionTaskWithChatPrompt, DEFAULT_GENERATE_KWARGS


class ContextAttributionTask(AttributionTaskWithChatPrompt):
    @property
    def target_token_range(self) -> Tuple[int, int]:
        return self.generation_token_start, self.generation_token_end


class SimpleContextAttributionTask(ContextAttributionTask):
    template = "Context: {context}\n\nQuery: {query}"

    def __init__(
        self,
        context: str,
        query: str,
        model: Any,
        tokenizer: Any,
        generate_kwargs: Dict[str, Any] = DEFAULT_GENERATE_KWARGS,
        previous_messages: Optional[List[Dict[str, str]]] = None,
        source_type: str = "token",
        cache: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Any],
    ):
        self.context = context
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
        return self.template.format(context=self.context, query=self.query)

    def _get_document_ranges(self):
        context_start = self.prompt.index(self.context)
        context_end = context_start + len(self.context)
        return [(context_start, context_end)]
