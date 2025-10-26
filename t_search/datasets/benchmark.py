
import inspect
from typing import Any, Callable, Literal, Optional

import torch

from .sampling import get_rand_points, lexsort


class Benchmark:

    def __init__(
        self,
        name: str | None,
        fn: Callable,
        train_sampling: Callable = get_rand_points,
        train_args: dict[str, Any] = None,
        test_sampling: Optional[Callable] = None,
        test_args: Optional[dict[str, Any]] = None,
    ):
        self.name = name
        if name is None:
            self.name = fn.__name__
        self.fn = fn
        self.train_sampling: Callable = train_sampling
        self.train_args: dict[str, Any] = {} if train_args is None else train_args
        self.test_sampling: Optional[Callable] = test_sampling
        self.test_args: Optional[dict[str, Any]] = test_args
        self.sampled = {}

    def with_train_sampling(self, train_sampling=None, **kwargs):
        return Benchmark(
            self.name, self.fn, (train_sampling if train_sampling is not None else self.train_sampling), kwargs
        )

    def with_test_sampling(self, test_sampling=None, **kwargs):
        return Benchmark(
            self.name,
            self.fn,
            self.train_sampling,
            self.train_args,
            (test_sampling if test_sampling is not None else self.test_sampling),
            kwargs,
        )

    def sample_set(
        self,
        set_name: Literal["train", "test"],
        device="cpu",
        dtype=torch.float32,
        generator: torch.Generator | None = None,
        sorted=False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if set_name in self.sampled:
            return self.sampled[set_name]
        if set_name == "test" and self.test_sampling is None:
            return self.sample_set("train", device)
        sample_args = self.train_args if set_name == "train" else self.test_args
        prepared_args = {
            k: (torch.tensor(v, device=device, dtype=dtype) if type(v) is list else v) for k, v in sample_args.items()
        }
        sample_fn = self.train_sampling if set_name == "train" else self.test_sampling
        signature = inspect.signature(sample_fn)
        if "generator" in signature.parameters:
            prepared_args["generator"] = generator
        free_vars = sample_fn(**prepared_args)
        gold_outputs = self.fn(*free_vars)
        if sorted:
            indices = lexsort(free_vars)
            new_free_vars = free_vars[:, indices]
            new_gold_outputs = gold_outputs[indices]
            del free_vars, gold_outputs
            free_vars = new_free_vars
            gold_outputs = new_gold_outputs
        self.sampled[set_name] = (free_vars, gold_outputs)
        return free_vars, gold_outputs
