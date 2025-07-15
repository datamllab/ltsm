import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

@dataclass
class TokenizerConfig:
    """
    This class holds all the configuration parameters to be used
    by ``ChronosTokenizer`` and ``ChronosModel``.
    """

    tokenizer_class: str
    tokenizer_kwargs: Dict[str, Any]
    n_tokens: int
    n_special_tokens: int
    pad_token_id: int
    eos_token_id: int
    use_eos_token: bool
    model_type: Literal["causal", "seq2seq"]
    context_length: int
    prediction_length: int
    num_samples: int
    temperature: float
    top_k: int
    top_p: float

    def __post_init__(self):
        assert (
            self.pad_token_id < self.n_special_tokens
            and self.eos_token_id < self.n_special_tokens
        ), f"Special token id's must be smaller than {self.n_special_tokens=}"

    def create_tokenizer(self) -> "ChronosTokenizer":
        if self.tokenizer_class == "MeanScaleUniformBins":
            return MeanScaleUniformBins(**self.tokenizer_kwargs, config=self)
        raise ValueError


class ChronosTokenizer:
    """
    A ``ChronosTokenizer`` definines how time series are mapped into token IDs
    and back.

    For details, see the ``input_transform`` and ``output_transform`` methods,
    which concrete classes must implement.
    """

    def input_transform(
        self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Turn a batch of time series into token IDs, attention map, and scale.

        Parameters
        ----------
        context
            A tensor shaped (batch_size, time_length), containing the
            timeseries to forecast. Use left-padding with ``torch.nan``
            to align time series of different lengths.

        Returns
        -------
        token_ids
            A tensor of integers, shaped (batch_size, time_length + 1)
            if ``config.use_eos_token`` and (batch_size, time_length)
            otherwise, containing token IDs for the input series.
        attention_mask
            A boolean tensor, same shape as ``token_ids``, indicating
            which input observations are not ``torch.nan`` (i.e. not
            missing nor padding).
        tokenizer_state
            An object that will be passed to ``output_transform``.
            Contains the relevant context to decode output samples into
            real values, such as location and scale parameters.
        """
        raise NotImplementedError()

    def output_transform(
        self, samples: torch.Tensor, tokenizer_state: Any
    ) -> torch.Tensor:
        """
        Turn a batch of sample token IDs into real values.

        Parameters
        ----------
        samples
            A tensor of integers, shaped (batch_size, num_samples, time_length),
            containing token IDs of sample trajectories.
        tokenizer_state
            An object returned by ``input_transform`` containing
            relevant context to decode samples, such as location and scale.
            The nature of this depends on the specific tokenizer.

        Returns
        -------
        forecasts
            A real tensor, shaped (batch_size, num_samples, time_length),
            containing forecasted sample paths.
        """
        raise NotImplementedError()


class MeanScaleUniformBins(ChronosTokenizer):
    def __init__(
        self, low_limit: float, high_limit: float, config: TokenizerConfig
    ) -> None:
        self.config = config
        self.centers = torch.linspace(
            low_limit,
            high_limit,
            config.n_tokens - config.n_special_tokens - 1,
        )
        self.boundaries = torch.concat(
            (
                torch.tensor([-1e20], device=self.centers.device),
                (self.centers[1:] + self.centers[:-1]) / 2,
                torch.tensor([1e20], device=self.centers.device),
            )
        )

    def input_transform(
        self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, length = context.shape

        if length > self.config.context_length:
            context = context[..., -self.config.context_length :]

        attention_mask = ~torch.isnan(context)
        scale = torch.nansum(
            torch.abs(context) * attention_mask, dim=-1
        ) / torch.nansum(attention_mask, dim=-1)
        scale[~(scale > 0)] = 1.0
        scaled_context = context / scale.unsqueeze(dim=-1)
        token_ids = (
            torch.bucketize(
                input=scaled_context,
                boundaries=self.boundaries,
                # buckets are open to the right, see:
                # https://pytorch.org/docs/2.1/generated/torch.bucketize.html#torch-bucketize
                right=True,
            )
            + self.config.n_special_tokens
        )
        token_ids[~attention_mask] = self.config.pad_token_id

        if self.config.use_eos_token:
            eos_tokens = torch.full(
                (batch_size, 1), fill_value=self.config.eos_token_id
            )
            token_ids = torch.concat((token_ids, eos_tokens), dim=1)
            eos_mask = torch.full((batch_size, 1), fill_value=True)
            attention_mask = torch.concat((attention_mask, eos_mask), dim=1)

        return token_ids, attention_mask, scale

    def output_transform(
        self, samples: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        
        scale_unsqueezed = scale.unsqueeze(-1).unsqueeze(-1)
        indices = torch.clamp(
            samples - self.config.n_special_tokens,
            min=0,
            max=len(self.centers) - 1,
        )
        self.centers = self.centers.to(samples.device)
        return self.centers[indices] * scale_unsqueezed

