import cvxpy as cvx
from cvxportfolio.expression import Expression
from .utils import values_in_time, null_checker

__all__ = ['ReturnsForecast', 'MPOReturnsForecast', 'MultipleReturnsForecasts']


class BaseReturnsModel(Expression):
    pass


class ReturnsForecast(BaseReturnsModel):
    """A single return forecast.

    Attributes:
      alpha_data: A dataframe of return estimates.
      delta_data: A confidence interval around the estimates.
      half_life: Number of days for alpha auto-correlation to halve.
    """

    def __init__(self, returns, delta=0., gamma_decay=None, name=None):
        null_checker(returns)
        self.returns = returns
        null_checker(delta)
        self.delta = delta
        self.gamma_decay = gamma_decay
        self.name = name

    def weight_expr(self, t, wplus=None, z=None, v=None, **kwargs):
        """Returns the estimated alpha.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        # 兼容旧版参数名 `w_plus`
        if wplus is None and 'w_plus' in kwargs:
          wplus = kwargs.get('w_plus')

        alpha = cvx.multiply(
          values_in_time(self.returns, t), wplus)
        alpha -= cvx.multiply(
            values_in_time(self.delta, t), cvx.abs(wplus))
        return cvx.sum(alpha)

    def weight_expr_ahead(self, t, tau, wplus=None, **kwargs):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """

        if wplus is None and 'w_plus' in kwargs:
          wplus = kwargs.get('w_plus')

        alpha = self.weight_expr(t, wplus)
        if tau > t and self.gamma_decay is not None:
            alpha *= (tau - t).days**(-self.gamma_decay)
        return alpha


class MPOReturnsForecast(BaseReturnsModel):
    """A single alpha estimation.

    Attributes:
      alpha_data: A dict of series of return estimates.
    """

    def __init__(self, alpha_data, gamma_decay=None): # <--- 加上这个参数
        self.alpha_data = alpha_data
        self.gamma_decay = gamma_decay

    def weight_expr_ahead(self, t, tau, wplus=None, **kwargs):
        if wplus is None and 'w_plus' in kwargs:
          wplus = kwargs.get('w_plus')
        
        # 1. 提取预测值
        alpha = self.alpha_data[(t, tau)].values.T * wplus
        
        # 2. 加入衰减逻辑 (仿照 ReturnsForecast)
        if tau > t and self.gamma_decay is not None:
            alpha *= (tau - t).days**(-self.gamma_decay)
            
        return alpha

class MultipleReturnsForecasts(BaseReturnsModel):
    """A weighted combination of alpha sources.

    Attributes:
      alpha_sources: a list of alpha sources.
      weights: An array of weights for the alpha sources.
    """

    def __init__(self, alpha_sources, weights):
        self.alpha_sources = alpha_sources
        self.weights = weights

    def weight_expr(self, t, wplus=None, z=None, v=None, **kwargs):
        """Returns the estimated alpha.

        Args:
            t: time estimate is made.
            wplus: An expression for holdings.
            tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        alpha = 0
        if wplus is None and 'w_plus' in kwargs:
          wplus = kwargs.get('w_plus')
        for idx, source in enumerate(self.alpha_sources):
          # 下面使用位置/兼容调用源模型
          alpha += source.weight_expr(t, wplus) * self.weights[idx]
        return alpha

    def weight_expr_ahead(self, t, tau, wplus=None, **kwargs):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        alpha = 0
        if wplus is None and 'w_plus' in kwargs:
          wplus = kwargs.get('w_plus')
        for idx, source in enumerate(self.alpha_sources):
          alpha += source.weight_expr_ahead(t, tau, wplus) * self.weights[idx]
        return alpha
