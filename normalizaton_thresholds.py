from dataclasses import dataclass

@dataclass
class NormalizationThresholds:
    norm_min: int
    norm_max: int
    norm_lower_bound: float
    norm_upper_bound: float