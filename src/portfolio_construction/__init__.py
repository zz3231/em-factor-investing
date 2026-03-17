from .equal_weight import compute_weights as equal_weight
from .inverse_variance import compute_weights as inverse_variance
from .min_variance import compute_weights as min_variance
from .max_sharpe import compute_weights as max_sharpe
from .risk_parity import compute_weights as risk_parity
from .hrp import compute_weights as hrp
from .momentum_weight import compute_weights as momentum_weight
from .black_litterman import compute_weights as black_litterman
from .mean_cvar import compute_weights as mean_cvar
from .turnover_penalized import compute_weights as turnover_penalized

ALL_METHODS = {
    'EqualWeight': equal_weight,
    'InverseVol': inverse_variance,
    'MinVariance': min_variance,
    'MaxSharpe': max_sharpe,
    'RiskParity': risk_parity,
    'HRP': hrp,
    'Momentum': momentum_weight,
    'BlackLitterman': black_litterman,
    'MeanCVaR': mean_cvar,
    'TO_MVO': turnover_penalized,
}

# Methods that use weight bounds
CONSTRAINED_METHODS = {
    'MinVariance', 'MaxSharpe', 'RiskParity',
    'BlackLitterman', 'MeanCVaR', 'TO_MVO',
}
