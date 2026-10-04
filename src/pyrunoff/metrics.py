from dataclasses import dataclass

import numpy
import pandas
from numpy.typing import ArrayLike, NDArray


@dataclass(slots=True)
class KgeResult:
    """Kling–Gupta Efficiency score and its component statistics.

    :param kge: Overall KGE′ score.
    :param corr_coef: Correlation between observed and simulated values.
    :param gamma: Ratio of coefficients of variation.
    :param beta: Ratio of means.
    """

    kge: float
    corr_coef: float
    gamma: float
    beta: float

    @classmethod
    def empty(cls) -> "KgeResult":
        """Return a result whose score and components are all undefined."""
        return KgeResult(numpy.nan, numpy.nan, numpy.nan, numpy.nan)


def nse(
    observed_data: pandas.Series,
    simulated_data: pandas.Series,
    use_low_flow_transform: bool = False,
    low_flow_percentile: float = 25,
) -> float:
    """
    Calculate the Nash–Sutcliffe Efficiency (NSE).

    :param observed_data: Observed values indexed by time.
    :param simulated_data: Simulated values indexed by time.
    :param use_low_flow_transform: Whether to apply the log-sinh transform
        before calculating the metric.
    :param low_flow_percentile: Percentile of positive observed values used as
        the transform scale. Must be between 0 and 100; the default is 25.
    :return: NSE value, or ``numpy.nan`` when observed values have zero variance.
    """

    obs, sim = _get_ts_arr(
        observed_data, simulated_data, use_low_flow_transform, low_flow_percentile
    )
    mean_obs = numpy.mean(obs)
    denominator = numpy.sum((obs - mean_obs) ** 2)
    if denominator == 0:
        return numpy.nan

    numerator = numpy.sum((obs - sim) ** 2)

    return 1 - numerator / denominator


def kge(
    observed_data: pandas.Series,
    simulated_data: pandas.Series,
    use_low_flow_transform: bool = False,
    low_flow_percentile: float = 25,
) -> KgeResult:
    """
    Calculate the 2012 Kling–Gupta Efficiency (KGE′) and its components.

    :param observed_data: Observed values indexed by time.
    :param simulated_data: Simulated values indexed by time.
    :param use_low_flow_transform: Whether to apply the log-sinh transform
        before calculating the metric.
    :param low_flow_percentile: Percentile of positive observed values used as
        the transform scale. Must be between 0 and 100; the default is 25.
    :return: KGE′ score and component statistics. Undefined results are
        represented by ``numpy.nan`` values.
    """
    obs, sim = _get_ts_arr(
        observed_data, simulated_data, use_low_flow_transform, low_flow_percentile
    )

    if len(obs) < 2:
        return KgeResult.empty()

    if sim.mean() == 0:
        corr_coef = numpy.nan
    else:
        corr_coef = numpy.corrcoef(obs, sim)[0, 1]

    mean_obs = numpy.mean(obs)
    std_obs = numpy.std(obs, ddof=1)

    if (
        numpy.isnan(corr_coef)
        or numpy.isclose(std_obs, 0)
        or numpy.isclose(mean_obs, 0)
    ):
        return KgeResult.empty()

    mean_sim = numpy.mean(sim)

    std_sim = numpy.std(sim, ddof=1)

    beta = mean_sim / mean_obs

    cv_obs = std_obs / mean_obs
    cv_sim = 0.0 if numpy.isclose(mean_sim, 0) else float(std_sim / mean_sim)
    gamma = cv_sim / cv_obs

    kge_value = 1 - numpy.sqrt(
        (corr_coef - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2
    )

    return KgeResult(
        kge=float(kge_value),
        corr_coef=float(corr_coef),
        gamma=float(gamma),
        beta=float(beta),
    )


def _log_sinh_transform(q: ArrayLike, alpha: float) -> NDArray[numpy.float64]:
    """
    Apply an inverse hyperbolic sine (log-sinh) transformation to flow values.

    This transformation behaves logarithmically for low values and linearly for
    high values, allowing low-flow behavior to be emphasized without distorting
    high-flow magnitudes.

    :param q:
        Input flow values (array-like).
    :param alpha:
        Scale parameter controlling the transition between logarithmic and linear
        behavior. Smaller values place greater emphasis on low flows.
    :return:
        NumPy array of transformed flow values with the same shape as the input.
    """
    return numpy.arcsinh(q / alpha)


def _get_ts_arr(
    observed_data: pandas.Series,
    simulated_data: pandas.Series,
    use_low_flow_transform: bool,
    low_flow_percentile: float,
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    """
    Align observed and simulated values and optionally transform low flows.

    :param observed_data: Observed values indexed by time.
    :param simulated_data: Simulated values indexed by time.
    :param use_low_flow_transform: Whether to apply the log-sinh transform.
    :param low_flow_percentile: Percentile of positive observed values used as
        the transform scale when the transform is enabled.
    :return: Observed and simulated arrays after pairwise removal of missing
        values and any requested transformation.
    """
    df = pandas.concat([observed_data, simulated_data], axis=1).dropna()
    df.columns = ["observed", "simulated"]

    observed_data = df["observed"].to_numpy(dtype=float)
    simulated_data = df["simulated"].to_numpy(dtype=float)

    if use_low_flow_transform:
        if low_flow_percentile <= 0 or low_flow_percentile >= 100:
            raise ValueError("low_flow_percentile must be between 0 and 100")

        eps = 1e-6
        pos = observed_data[observed_data > 0]
        if len(pos) == 0:
            alpha = eps
        else:
            alpha = max(numpy.percentile(pos, low_flow_percentile), eps)

        observed_data = _log_sinh_transform(observed_data, alpha=alpha)
        simulated_data = _log_sinh_transform(simulated_data, alpha=alpha)

    return observed_data, simulated_data
