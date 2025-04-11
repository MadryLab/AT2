from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch as ch
from tqdm.auto import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq


from ..utils import split_text


DEFAULT_GENERATE_KWARGS = {
    "max_new_tokens": 512,
    "do_sample": False,
    "top_p": None,
    "top_k": None,
    "temperature": None,
}


def create_mask(num_sources, alpha, seed):
    random = np.random.RandomState(seed)
    return random.choice([False, True], size=num_sources, p=[1 - alpha, alpha])


def create_masks(num_masks, num_sources, alpha, seed):
    masks = np.zeros((num_masks, num_sources), dtype=bool)
    for mask_index in range(num_masks):
        masks[mask_index] = create_mask(num_sources, alpha, seed + mask_index)
    return masks


def compute_logit_probs(logits, labels):
    batch_size, seq_length, num_classes = logits.shape
    reshaped_logits = logits.reshape(batch_size * seq_length, num_classes)
    reshaped_labels = labels.reshape(batch_size * seq_length)
    correct_logits = reshaped_logits.gather(-1, reshaped_labels[:, None])[:, 0]
    cloned_logits = reshaped_logits.clone()
    cloned_logits.scatter_(-1, reshaped_labels[:, None], -ch.inf)
    other_logits = cloned_logits.logsumexp(dim=-1)
    reshaped_outputs = correct_logits - other_logits
    return reshaped_outputs.reshape(batch_size, seq_length)


class AttributionTask(ABC):
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        generate_kwargs: Dict[str, Any] = DEFAULT_GENERATE_KWARGS,
        cache: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generate_kwargs: Dict[str, Any] = generate_kwargs
        self._cache: Dict[str, Any] = cache or {}

    @property
    @abstractmethod
    def input_text(self):
        """The relevant input text for this task (e.g., a context and query)."""

    def get_input_tokens(self, return_tensors=None):
        if "input_tokens" not in self._cache:
            self._cache["input_tokens"] = {}
        if return_tensors not in self._cache["input_tokens"]:
            self._cache["input_tokens"][return_tensors] = self.tokenizer(
                self.input_text, add_special_tokens=False, return_tensors=return_tensors
            )

        return self._cache["input_tokens"][return_tensors]

    def get_input_text_and_tokens(self, return_tensors=None, mask=None):
        input_text = self.input_text
        input_tokens = self.get_input_tokens(return_tensors=return_tensors)
        if mask is not None:
            assert return_tensors == "pt"
            for i, (start, end) in enumerate(self.source_token_ranges):
                input_tokens["attention_mask"][0][start:end] = int(mask[i])
        return input_text, input_tokens

    def set_generation(self, generation):
        input_text, _ = self.get_input_text_and_tokens()
        self._cache["text"] = input_text + generation

    def _format_generation_hidden_states(self, output):
        if output.hidden_states is None:
            return None
        hidden_states = output.hidden_states
        num_layers = len(hidden_states[0])
        num_iterations = len(hidden_states)
        formatted_hidden_states = []
        for layer in range(num_layers):
            cur_hidden_states = ch.cat(
                [hidden_states[i][layer] for i in range(num_iterations)], dim=1
            )
            formatted_hidden_states.append(cur_hidden_states)
        return formatted_hidden_states

    def generate(self, mask=None, output_hidden_states=False):
        input_text, input_tokens = self.get_input_text_and_tokens(
            return_tensors="pt", mask=mask
        )
        output = self.model.generate(
            **input_tokens.to(self.model.device),
            **self.generate_kwargs,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=True,
        )
        hidden_states = self._format_generation_hidden_states(output)
        # We take the original input because sometimes encoding and decoding changes it
        raw_text = self.tokenizer.decode(output.sequences[0])
        input_ids = input_tokens["input_ids"][0]
        input_length = len(self.tokenizer.decode(input_ids))
        generation = raw_text[input_length:]
        return {
            "text": input_text + generation,
            "hidden_states": hidden_states,
        }

    def get_hidden_states(self):
        batch = self.get_tokens(return_tensors="pt").to(self.model.device)
        with ch.no_grad():
            output = self.model(**batch, output_hidden_states=True)
        hidden_states = [
            output.hidden_states[layer][:, :-1]
            for layer in range(len(output.hidden_states))
        ]
        return hidden_states

    @classmethod
    def batch_generate(cls, tasks: List["AttributionTask"]):
        model = tasks[0].model
        tokenizer = tasks[0].tokenizer
        inputs_texts = []
        inputs_tokens = []
        for task in tasks:
            input_text, input_tokens = task.get_input_text_and_tokens()
            inputs_texts.append(input_text)
            inputs_tokens.append(input_tokens)
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
        batch = collator(inputs_tokens).to(model.device)
        output_ids = model.generate(
            **batch,
            **tasks[0].generate_kwargs,
        )
        eos_token_ids = ch.tensor(
            model.generation_config.eos_token_id,
            device=output_ids.device,
        )
        generations = []
        for example_index in range(len(inputs_tokens)):
            input_length = len(inputs_tokens[example_index]["input_ids"])
            start_index = batch["input_ids"].shape[1] - input_length
            end_indices = ch.where(
                ch.isin(
                    output_ids[example_index][start_index + input_length :],
                    eos_token_ids,
                )
            )[0]
            end_index = (
                end_indices[0] + start_index + input_length + 1
                if len(end_indices) > 0
                else output_ids.shape[1]
            )
            cur_output_ids = output_ids[example_index][start_index:end_index]
            raw_text = tokenizer.decode(cur_output_ids)
            input_length = len(
                tokenizer.decode(inputs_tokens[example_index]["input_ids"])
            )
            generation = raw_text[input_length:]
            generations.append(generation)
            tasks[example_index].set_generation(generation)
        return generations

    @property
    def text(self):
        if self._cache.get("text") is None:
            generate_outputs = self.generate(output_hidden_states=True)
            self._cache.update(generate_outputs)
        return self._cache["text"]

    @property
    def hidden_states(self):
        if self._cache.get("hidden_states") is None:
            if "text" in self._cache:
                self._cache["hidden_states"] = self.get_hidden_states()
            else:
                generate_outputs = self.generate(output_hidden_states=True)
                self._cache.update(generate_outputs)
        return self._cache["hidden_states"]

    def get_tokens(self, return_tensors=None):
        if "tokens" not in self._cache:
            self._cache["tokens"] = {}
        if return_tensors not in self._cache["tokens"]:
            self._cache["tokens"][return_tensors] = self.tokenizer(
                self.text, add_special_tokens=False, return_tensors=return_tensors
            )

        return self._cache["tokens"][return_tensors]

    @property
    def generation_token_start(self):
        _, input_tokens = self.get_input_text_and_tokens()
        return len(input_tokens["input_ids"])

    @property
    def generation_token_end(self):
        eos_token_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        tokens = self.get_tokens()
        if tokens["input_ids"][-1] in eos_token_ids:
            return len(tokens["input_ids"]) - 1
        else:
            return len(tokens["input_ids"])

    @property
    def generation(self):
        tokens = self.get_tokens()
        generation_start = tokens.token_to_chars(self.generation_token_start).start
        generation_end = tokens.token_to_chars(self.generation_token_end - 1).end
        return self.text[generation_start:generation_end]

    @property
    def source_token_ranges(self):
        if self._cache.get("source_token_ranges") is None:
            self._cache["source_token_ranges"] = self._get_source_token_ranges()
        return self._cache["source_token_ranges"]

    @abstractmethod
    def _get_source_token_ranges(self):
        """Get the token ranges for the sources."""

    @property
    def num_sources(self):
        return len(self.source_token_ranges)

    def get_source(self, index):
        tokens = self.get_tokens()
        token_start, token_end = self.source_token_ranges[index]
        start = tokens.token_to_chars(token_start).start
        end = tokens.token_to_chars(token_end - 1).end
        return self.text[start:end]

    @property
    def sources(self):
        return [self.get_source(i) for i in range(self.num_sources)]

    @property
    @abstractmethod
    def target_token_range(self):
        """Get the token range for the target."""

    @property
    def has_empty_target(self):
        target_token_start, target_token_end = self.target_token_range
        return target_token_start == target_token_end

    def get_sub_token_ranges(
        self,
        token_range: Tuple[int, int],
        split_by: Optional[str] = None,
    ):
        token_start, token_end = token_range
        if token_start == token_end:
            return []
        if split_by is None:
            ranges = [token_range]
        elif split_by == "token":
            ranges = [(start, start + 1) for start in range(*token_range)]
        elif split_by == "word" or split_by == "sentence":
            tokens = self.get_tokens()
            start = tokens.token_to_chars(token_start).start
            end = tokens.token_to_chars(token_end - 1).end
            text = self.text[start:end]
            _, _, indices = split_text(text, split_by)
            ranges = []
            for cur_start, cur_end in indices:
                cur_start_token = tokens.char_to_token(start + cur_start)
                cur_end_token = tokens.char_to_token(start + cur_end - 1) + 1
                ranges.append((cur_start_token, cur_end_token))
        return ranges

    def get_sub_target_token_ranges(
        self, split_by: Optional[str] = None, relative: bool = False
    ):
        ranges = self.get_sub_token_ranges(self.target_token_range, split_by)
        if relative:
            offset, _ = self.target_token_range
            ranges = [(start - offset, end - offset) for start, end in ranges]
        return ranges

    def get_ablated_inputs(self, mask=None, ablation_method="mask"):
        tokens = self.get_tokens()
        attention_mask = np.ones(len(tokens["input_ids"]), dtype=np.int32)
        if mask is not None:
            for i, (start, end) in enumerate(self.source_token_ranges):
                attention_mask[start:end] = mask[i]
        if ablation_method == "mask":
            inputs = {
                "input_ids": tokens["input_ids"],
                "attention_mask": attention_mask.tolist(),
                "labels": tokens["input_ids"],
            }
        elif ablation_method == "remove":
            input_ids = [
                token
                for token, include in zip(tokens["input_ids"], attention_mask)
                if include
            ]
            inputs = {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": input_ids,
            }
        else:
            raise ValueError(f"Invalid ablation method: {ablation_method}")
        return inputs

    @property
    def target(self):
        target_token_start, target_token_end = self.target_token_range
        if target_token_start == self.generation_token_end:
            return ""
        tokens = self.get_tokens()
        start = tokens.token_to_chars(target_token_start).start
        end = tokens.token_to_chars(target_token_end - 1).end
        return self.text[start:end]

    @property
    def target_ids(self):
        target_token_start, target_token_end = self.target_token_range
        tokens = self.get_tokens()
        return tokens["input_ids"][target_token_start:target_token_end]

    def show_target_with_indices(self, split_by: str = "sentence"):
        parts, separators, start_indices = split_text(self.target, split_by)
        formatted_words = []

        RED = "\033[36m"
        RESET = "\033[0m"

        for word, idx in zip(parts, start_indices):
            formatted_words.append(f"{RED}[{idx}]{RESET}{word}")

        result = "".join(sep + word for sep, word in zip(separators, formatted_words))
        print(result)

    def target_range_to_token_range(
        self, start_index=None, end_index=None, relative=False
    ):
        target_token_start, target_token_end = self.target_token_range
        tokens = self.get_tokens()
        offset = tokens.token_to_chars(target_token_start).start
        token_start = (
            tokens.char_to_token(start_index + offset)
            if start_index is not None
            else target_token_start
        )
        token_end = (
            tokens.char_to_token(end_index + offset - 1) + 1
            if end_index is not None
            else target_token_end
        )
        if relative:
            token_start -= target_token_start
            token_end -= target_token_start
        return token_start, token_end

    def create_ablation_dataset(
        self, num_masks, alpha, base_seed=0, ablation_method="mask"
    ):
        masks = create_masks(num_masks, self.num_sources, alpha, base_seed)
        dataset_dict = defaultdict(list)
        for mask in masks:
            ablated_inputs = self.get_ablated_inputs(
                mask=mask, ablation_method=ablation_method
            )
            for key, value in ablated_inputs.items():
                dataset_dict[key].append(value)
        return masks, Dataset.from_dict(dataset_dict)

    def make_ablation_loader(self, dataset, batch_size):
        collate_fn = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding="longest")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        return loader

    def get_target_logit_probs(self, batch, enable_grad=False):
        if enable_grad:
            output = self.model(**batch)
        else:
            with ch.no_grad():
                output = self.model(**batch)
        target_token_start, target_token_end = self.target_token_range
        logits = output.logits[:, target_token_start - 1 : target_token_end - 1]
        labels = batch["labels"][:, target_token_start:target_token_end]
        return compute_logit_probs(logits, labels)

    def get_masks_and_logit_probs(
        self,
        num_masks,
        alpha,
        batch_size,
        base_seed=0,
        ablation_method="mask",
        verbose=False,
    ):
        masks, dataset = self.create_ablation_dataset(
            num_masks,
            alpha,
            base_seed=base_seed,
            ablation_method=ablation_method,
        )
        loader = self.make_ablation_loader(dataset, batch_size)
        target_start, target_end = self.target_token_range
        logit_probs = ch.zeros(
            (len(dataset), target_end - target_start), device=self.model.device
        )

        start_index = 0
        for batch in tqdm(loader, disable=not verbose):
            batch = {key: value.to(self.model.device) for key, value in batch.items()}
            cur_logit_probs = self.get_target_logit_probs(batch)
            cur_batch_size, _ = batch["input_ids"].shape
            logit_probs[start_index : start_index + cur_batch_size] = cur_logit_probs
            start_index += cur_batch_size

        return masks, logit_probs.cpu().type(ch.float32).numpy()
