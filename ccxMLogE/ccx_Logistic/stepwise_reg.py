import pandas as pd
import statsmodels.api as sm


# x-变量的DataFrame
# 后向选择
def back_reg(x, y):
    logit = sm.Logit(y, x)  # intercept included
    result = logit.fit()
    pvalues = result.pvalues[:-1]
    var = pvalues.idxmax()
    if pvalues.loc[var] > 0.05:
        # var=pvalues.index[pvalues<0.05]
        return back_reg(x.drop(var, axis=1), y)
    else:
        return result
        # return result


# 前向选择
# x0:截距项；cols:自变量名称列表；y：y值
def fwd_reg(x0, x, cols, y):
    s = []
    for var in cols:
        logit = sm.Logit(y, pd.concat([x0, x[var]], axis=1))  # intercept included
        result = logit.fit()
        pvalues = result.pvalues
        s.append((pvalues[var], var))
    p = min(s)[0]
    p_name = min(s)[1]

    if p < 0.05:

        x0 = pd.concat([x0, x[p_name]], axis=1)
        cols.remove(p_name)
        return fwd_reg(x0, x, cols, y)
    else:

        logit = sm.Logit(y, x0)
        result = logit.fit()
        return result


# 双向选择（逐步回归）

def step_reg(x, cols, y):
    x0 = x.intercept
    s = []
    for var in cols:
        logit = sm.Logit(y, pd.concat([x0, x[var]], axis=1))  # intercept included
        result = logit.fit()
        pvalues = result.pvalues
        s.append((pvalues[var], var))
    p = min(s)[0]
    p_name = min(s)[1]

    if p < 0.05:
        x0 = pd.concat([x0, x[p_name]], axis=1)
        cols.remove(p_name)

        res = back_reg(x0, y)  # 后向选择
        x0 = x0[res.pvalues.index]
        return fwd_reg(x0, x, cols, y)
    else:
        logit = sm.Logit(y, x0)
        result = logit.fit()
        return result