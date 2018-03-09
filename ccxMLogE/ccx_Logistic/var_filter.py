from sklearn.model_selection import train_test_split
from ccxMLogE.ccx_Logistic.cut_bins import woe_var_get
from ccxMLogE.ccx_Logistic.vif_filter import filter_vif
from ccxMLogE.ccx_Logistic.stepwise_reg import back_reg,fwd_reg,step_reg
import pandas as pd,pickle,os

def data_split(data,y,split_ratio=0.3):
    train_data,test_data,train_target,test_target = train_test_split(data,y,random_state=0,test_size=split_ratio)
    return train_data,train_target,test_data,test_target

def woeTransData(path,data,y,var_set,method,isdump,flag):
    var2woe,iv_get = woe_var_get(path,data,var_set,y,method,isdump=isdump,binClass=flag)
    with open(os.path.join(path,'pkl/var2woe.pkl'),'wb') as f: pickle.dump(var2woe,f)
    with open(os.path.join(path,'pkl/iv_get.pkl'),'wb') as f: pickle.dump(iv_get,f)

    return iv_get
class Varfilter():
    '''
    iv 筛选变量
    '''
    def __init__(self,threshold=0.02):
        self.threshold=0.02

    def filterIV(self,iv_get,var2woe):
        dict_filt = {k:v for k,v in iv_get.items() if v<self.threshold}
        var2woe.columns = list(map(lambda i: i[4:], var2woe.columns))
        var2woe = var2woe[list(set(var2woe.columns).difference(set(dict_filt.keys())))]  # 过滤掉iv值小于thres的变量
        return var2woe, dict_filt


def filterMultiCol(var2woe,thres):
    '''

    :param data:
    :param var_set:
    :return:  vif 去重
    '''

    vif_f = filter_vif(thres)
    var_vifilter = vif_f.calculate_vif(var2woe)
    return var_vifilter

def filterStep(data,y,type):
    '''

    :param data:
    :param y: 标签
    :param type: 逐步回归类型
    :return: 逐步回归筛选后的变量
    '''
    # type = 1
    if type == 1: # 后向逐步回归
        data2 = data.copy()
        data2['intercept'] = 1
        result = back_reg(data2, y)

    elif type == 2: # 前向逐步回归
        cols = list(data.columns)
        data2 = data.copy()
        data2['intercept'] = 1
        x0 = data2.intercept
        result = fwd_reg(x0, data2, cols, y)
    elif type == 3: # 双向逐步回归
        cols = list(data.columns)
        data2 = data.copy()
        data2['intercept'] = 1
        result = step_reg(data2, cols, y)
    step_res = pd.concat([result.params.rename('params'),result.pvalues.rename('pvalue')],axis=1)
    step_var = data[step_res.index[step_res.index!='intercept']]
    return step_res,step_var







