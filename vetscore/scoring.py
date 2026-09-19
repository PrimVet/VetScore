from vetscore import config


class WeightedScoring:
    """Computes harm-potential-weighted verification scores for a set of claims."""

    def __init__(self, harm_pow: float = config.HARM_POW):
        self.harm_pow = harm_pow

    def weight(self, harm_potential: int | float) -> float:
        return float(harm_potential) ** self.harm_pow

    def __call__(self, claims: list[dict]) -> float:
        if not claims:
            return 1.0

        weights = [self.weight(c["harm_potential"] or 0) for c in claims]
        total = sum(weights)

        if total == 0:
            supported_count = sum(1 for c in claims if c["is_supported"])
            return supported_count / len(claims)

        unsupported_sum = sum(
            w for c, w in zip(claims, weights) if not c["is_supported"]
        )
        penalty = unsupported_sum / total
        return max(0.0, min(1.0, 1.0 - penalty))
