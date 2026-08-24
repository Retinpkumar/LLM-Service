# Pricing per million tokens (MTok), in USD.
# Source: https://www.anthropic.com/pricing — verify before updating.
PRICING = {
    "claude-sonnet-4-6": {
        "input_per_mtok": 3.00,
        "output_per_mtok": 15.00,
    },
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICING:
        raise ValueError(f"No pricing data for model: {model}")
    rates = PRICING[model]
    input_cost = (input_tokens / 1_000_000) * rates["input_per_mtok"]
    output_cost = (output_tokens / 1_000_000) * rates["output_per_mtok"]
    return input_cost + output_cost
