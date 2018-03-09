from sklearn.linear_model import LogisticRegression as LR
import pandas as pd
import numpy as np
import pickle,os
from ccxMLogE.ccx_Logistic.cut_bins import get_woe_class,get_woe_num
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
def lr_fit(modelpath,x,y,k=0):
    '''

    :param x: train_data
    :param y: train_target
    :return: logisticReg model,筛选完变量后的数据
    '''
    x['intercept'] = 1
    logit = sm.Logit(y, x)  # intercept included
    result = logit.fit()
    # clf = LR(C=1e9)
    # clf.fit(x,y)
    # df = clf.coef_[0]
    df = result.params.drop('intercept')
    if any(np.asarray(df) < 0):
        print ('系数出现负数，执行第{}次剔除'.format(k))
        k=k+1
        loc = list(df).index(min(df))
        col = list(df.index)
        col.remove(col[loc])
        return lr_fit(modelpath, x[col], y, k)
    else:
        # with open(os.path.join(modelpath,'model/var_model.pkl'),'wb') as f: pickle.dump(list(df.index),f) # 保存变量结果
        with open(os.path.join(modelpath, 'model/LR_model.pkl'), 'wb') as f:
            pickle.dump(result, f)  # 保存模型
        df_res = pd.concat([result.params, result.pvalues, result.tvalues], axis=1)
        df_res.columns = ['coefs', 'pvalues', 'tvalues']
        df_res.sort_values(by='coefs', ascending=False).to_csv(os.path.join(modelpath, 'model/var_summary.csv'))
        return result, x, df_res

def testdata_tf(test_data,vartype,fillData):
    '''

    :param test_data:
    :return: 变量筛选完以及woe转换的测试数据
    '''

    number_na,str_na,missVar = fillData.load_num()
    with open('var_model.pkl','rb') as f: var = pickle.load(f)

    filter_data = test_data[var]
    var_num = list(filter_data.columns.intersection(set(vartype[1])))
    var_class = list(filter_data.columns.intersection(set(vartype[0])))
    num = number_na.index.intersection(set(var_num))
    filter_data[list(num)] = filter_data[list(num)].fillna(number_na.loc[list(num)]) # 数值型变量的填充
    missVar2 = set(missVar).intersection(set(var))
    filter_data[list(missVar2)] = filter_data[list(missVar2)].fillna('missing') # missing 变量填充

    class_var = str_na.index.intersection(set(var_class))
    filter_data[list(class_var)] = filter_data[list(class_var)].fillna(str_na.loc[list(class_var)])
    woe_num = pd.concat(map(lambda i: get_woe_num(i,filter_data),var_num),axis=1)
    woe_class = pd.concat(map(lambda i: get_woe_class(i,filter_data),var_class),axis=1)
    woe_res = pd.concat([woe_num,woe_class],axis=1)
    return woe_res



def get_AB(baseScore,pdo):
    '''
    由基准分和PDO得到A,B系数
    :param baseScore:
    :param pdo:
    :return: A,B
    '''
    A = baseScore
    B = pdo/np.log(2)
    return A,B


def scoreCardOut(path, clf, A, B):
    '''
    输出评分卡
    :param clf:
    :param var:
    :param A:
    :param B:
    :return: 评分卡输出
    '''
    # varList = var
    coef = clf.params
    # coef_dict = dict(zip(varList,list(coef[0])))
    coef_dict = coef.to_dict()
    _ = coef_dict.pop('intercept')
    varList = list(coef_dict.keys())
    intercept = clf.params.loc['intercept']
    df = pd.DataFrame([['基础分',None,A-B*intercept]],columns=['变量','条件区间','分值'])
    for i in varList:
        with open(os.path.join(path,'pkl/{}_woedict.pkl'.format(i)),'rb') as f: woe_res = pickle.load(f)
        woe_res.pop('All')
        df_woe = pd.DataFrame(pd.Series(woe_res)).reset_index()
        df_woe.columns = ['条件区间','分值']
        df_woe['分值'] = -df_woe['分值']*B*coef_dict[i]
        df_woe['变量'] = i
        df_woe = df_woe[df.columns]
        df = df.append(df_woe)

    return df

def read_savedata(modelpath):
    with open(os.path.join(modelpath, 'pkl/var2woe.pkl'), 'rb') as f:
        var2woe = pickle.load(f)  # 读取训练标签
    with open(os.path.join(modelpath, 'pkl/iv_get.pkl'), 'rb') as f:
        iv = pickle.load(f)  # 读取训练标签
    with open(os.path.join(modelpath, 'pkl/target.pkl'), 'rb') as f:
        train_target = pickle.load(f)  # 读取训练标签
    with open(os.path.join(modelpath, 'pkl/test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)  # 读取训练标签
    with open(os.path.join(modelpath, 'pkl/test_target.pkl'), 'rb') as f:
        test_target = pickle.load(f)  # 读取训练标签

    return var2woe,iv,train_target,test_data,test_target




