import numpy as np
import pandas as pd
from inspect import getmembers
from sklearn import tree
import pickle,os
from ccxMLogE.ccx_Logistic.discretization import Discretization
from ccxMLogE.ccx_Logistic.classVarMerge import classMerge, var_rank
################### iv值的计算和woe的转换
from ccxMLogE.ccx_Logistic.logLogis import ABS_log


def decis_bin(x,y):
    '''
    通过决策树分箱计算分割点
    :param x:
    :param y:
    :return:
    '''
    clf = tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=0.05)
    x=np.array(x).reshape(len(x),1)
    model = clf.fit(x, y)
    v_tree=getmembers( model.tree_ )
    v_tree_thres=dict(v_tree)['threshold']
    v_tree_thres=sorted(list(v_tree_thres[v_tree_thres!=-2]))
    split_p=[min(x)[0]]+v_tree_thres+[max(x)[0]+1] # 左闭右开
    return split_p

def get_bin(x,flag=0):
    '''
    获取数值区间的边界点,0 代表左边界，1代表右边界
    :param x:
    :return:
    '''
    x1_split = x.split(',')[flag]
    x1_split = float(x1_split.strip('()[]'))
    return x1_split

def IV_compute(var_cut2,y):

    crtab = pd.crosstab(var_cut2, y, margins=True)

    crtab.columns = ['good', 'bad', 'total']
    crtab['factor_per'] = crtab['total'] / len(y)
    crtab['bad_per'] = crtab['bad'] / crtab['total']
    crtab['p'] = crtab['bad'] / crtab.ix['All', 'bad']
    crtab['q'] = crtab['good'] / crtab.ix['All', 'good']
    crtab['woe'] = np.log((crtab['p']+1e-10) / (crtab['q']+1e-10))

    crtab['ID'] = np.arange(len(crtab))
    return crtab
@ABS_log('IV_numeric')
def IV_numeric(path,data,y,missingValue,bins=[],isdump=False):
    '''
    连续变量分箱
    :param data:  series
    :param col: 变量名
    :param missingValue: 分割点
    :param y: 数据的y值
    :param bins: 分割点
    :return: iv值和woe转换后的变量
    '''
    iv={}
    var_cut2 = pd.cut(data, bins, right=False)
    var_cut2 = var_cut2.append(missingValue)

    bin_var = IV_compute(var_cut2,y)
    if any(abs(bin_var.woe) == np.inf): # 如果woe出现inf值
        print ('invalid woe value encountered')
        loc_index = list(bin_var[abs(bin_var.woe)==np.inf].index.values) # 得到woe值为inf的索引值
        loc_index2 = list(bin_var[abs(bin_var.woe)==np.inf].ID) # 得到 woe值为inf的id号

        for i,j in zip(loc_index,loc_index2):
            if j!=0: # 非第一个索引
                points_drop = get_bin(i)  # 得到出错的边界点
                bins.remove(points_drop)
            else:
                points_drop = get_bin(i,flag=1)
                bins.remove(points_drop)


        return IV_numeric(data,y,missingValue,bins) # 递归计算IV
    else:
        col=data.name
        fpath = os.path.join(path,'ivRes/ivRes_%s.csv'%col)
        bin_var.to_csv(fpath) # 保存iv结果，便于可视化
        bin_var.rename(columns={'woe':'woe_%s'%col},inplace=True)
        woe_col = pd.merge(var_cut2.reset_index(),bin_var.reset_index(),on=col,how='left').set_index('id')['woe_%s'%col]
        iv[col] = sum((bin_var['p'] - bin_var['q']) * np.log(bin_var['p'] / bin_var['q']))

        if isdump:
            dict_woe = bin_var['woe_%s'%col].to_dict()
            f_woepath = os.path.join(path,'./pkl/%s_woedict.pkl' % col)
            with open(f_woepath, 'wb') as f: pickle.dump(dict_woe, f)
        return iv,woe_col

@ABS_log('IV_class')
def IV_class(path,data,y,isdump,flag=0):
    '''
    离散变量分箱： 少于10个的直接分箱，多于10个的先合并再分箱
    :param data: series
    :param col: 变量
    :param y: y值
    :param isdump: 是否需要保存
    :param flag: 0、是否无论离散变量取值多少，都直接分箱；1、聚类合并分箱
    :return: iv and woe转换后的变量
    '''
    iv={}
    # 离散型变量 iv计算
    if (data.nunique()<=10 and flag!=0) or (flag==0):
        bin_var = IV_compute(data, y)
    elif data.nunique() > 10 and flag == 1:  # 聚类合并
        reclass_var = classMerge(data,y) # 合并分类变量
        N_var = data.apply(lambda i: reclass_var[i])
        bin_var = IV_compute(N_var, y)
        f_classdict = os.path.join(path,'classdict/%s_classdict.pkl' % data.name)
        with open(f_classdict, 'wb') as f:
            pickle.dump(reclass_var, f)  # 保留分类变量合并取值
    elif data.nunique() > 10 and flag == 2:  # 按违约率合并
        concat_data = pd.concat([data, y], axis=1)
        reclass_var = var_rank(concat_data, data.name, y.name)  # 合并分类变量
        N_var = data.apply(lambda i: reclass_var[i])
        bin_var = IV_compute(N_var, y)
        f_classdict = os.path.join(path, 'classdict/%s_classdict.pkl' % data.name)
        with open(f_classdict, 'wb') as f:
            pickle.dump(reclass_var, f)  # 保留分类变量合并取值
    col=data.name
    f_ivpath = os.path.join(path,'ivRes/ivRes_%s.csv'%col)
    bin_var.to_csv(f_ivpath) # 保存变量结果
    bin_var.rename(columns={'woe': 'woe_%s' % col}, inplace=True)
    try:
        woe_col = pd.merge(N_var.reset_index(), bin_var.reset_index(), on=col, how='left').set_index('id')[
            'woe_%s' % col]
    except:
        woe_col = pd.merge(data.reset_index(), bin_var.reset_index(), on=col, how='left').set_index('id')[
            'woe_%s' % col]
    iv[col] = sum((bin_var['p'] - bin_var['q']) * bin_var['woe_%s' % col])
    if isdump:
        dict_woe = bin_var['woe_%s'%col].to_dict()
        f_woepath = os.path.join(path, 'pkl/%s_woedict.pkl'%col)
        with open(f_woepath,'wb') as f: pickle.dump(dict_woe,f)  # 保留分类变量woe映射字典
    return iv,woe_col



def get_iv_woe_num(path,i,fill_data,y,method,isdump=False):
    '''
    连续型变量得到iv值和woe序列
    :param i: 变量名
    :param fill_data: 训练数据
    :param y: 目标变量
    :param method: 分箱方法
    :param isdump:是否存储
    :return: iv，woe转换结果
    '''

    var = fill_data[i]
    try:
        var2 = var[var!='missing'] # 剥离出变量值不为missing的样本
        if method == 1: # 决策树分箱
            bins = decis_bin(var2, y[var != 'missing'].tolist())
        elif method == 2: # 卡方分箱
            dis = Discretization(method='chimerge', max_intervals=5)
            bins = dis.fit(var2,y[var != 'missing'].tolist())
        missValue = var[var=='missing']
    except Exception as e:
        var_str = var.astype(str)
        var2 = var[var_str!='missing']
        if method == 1:
            bins = decis_bin(var2, y[var_str != 'missing'].tolist())
        elif method == 2:
            bins = dis.fit(var2, y[var_str != 'missing'].tolist())
        missValue = var[var_str == 'missing']
    iv,woe_res = IV_numeric(path,var2,y,missValue,bins,isdump=isdump)
    return iv,woe_res




#for i in set(type_data[1]).union(set(type_data[2])).difference(set(['flag_person'])):
# var_class = set(type_data[1]).union(set(type_data[2])).difference(set(['flag_person']))
def get_iv_woe_class(path,i,fill_data,y,isdump,flag=0):
    #分类型变量得到iv值和woe序列
    iv,woe = IV_class(path,fill_data[i],y,isdump,flag)
    return iv,woe
@ABS_log('woe_tran')
def woe_var_get(path,fill_data,var_set,y,binNum,isdump=False,binClass=0):
    '''

    :param fill_data:
    :param var_set:
    :param y: 目标变量
    :param binNum: 数值变量分箱方法
    :param binClass: 离散变量分箱方法
    :return: 合并后的woe变量
    '''
    # ans = map(lambda i: get_iv_woe_num(path, i, fill_data, y, binNum, isdump=isdump), var_set[1])
    iv={}
    df_num = pd.DataFrame()
    for i in var_set[1]:
        bin_res = get_iv_woe_num(path,i,fill_data,y,binNum,isdump=True)
        df_num = pd.concat([df_num,bin_res[1]],axis=1)
        iv.update(bin_res[0])
    # var_num = pd.concat(
    #     map(lambda i: get_iv_woe_num(path,i, fill_data,y,binNum,isdump=isdump)[1], var_set[1]), axis=1)
    # var_class = pd.concat(map(lambda i: get_iv_woe_class(path,i,fill_data,y,isdump=isdump,flag=binClass)[1],var_set[0]),axis=1)
    df_class = pd.DataFrame()
    for i in var_set[0]:
        bin_res = get_iv_woe_class(path, i, fill_data, y, isdump=isdump,flag=binClass)
        df_class = pd.concat([df_class, bin_res[1]], axis=1)
        iv.update(bin_res[0])
    VarWoe = pd.concat([df_class,df_num],axis=1)
    with open(os.path.join(path,'ivRes/iv_dict.pkl'),'wb') as f: pickle.dump(iv,f)
    return VarWoe,iv


#################################以下是对测试集数据进行映射


def get_category_num(x,dict_var):
 # 将数值映射到category
    dict_copy = dict_var.copy()
    if 'missing' in dict_copy.keys():
        dict_copy.pop('missing')
 # for i in dict_copy.keys():
 #     num = [k.strip('[]()') for k in i.split(',')]
 #     if x>=float(num[0]) and x<float(num[1]):
 #         trans_woe = dict_copy[i]
 #         break
 # python 3.6 与3.5相比，多了个interval类型的数据
    for i in dict_copy.keys():
        if x in i:
            trans_woe = dict_copy[i]
            break
    return trans_woe

def get_category_class(x,dict_var):
 # 将类别值映射到category
    if x in dict_var.keys():
        trans_woe = dict_var[x]
    else:
        trans_woe = np.nan
    return trans_woe


# for i in set(Data1.columns).difference(set(['id','flag_person'])):
def get_woe_num(path,var,data):
    # 用于预测集数值型变量woe转换
    with open(os.path.join(path,'pkl/%s_woedict.pkl' % var), 'rb') as f: dict_var = pickle.load(f)
    dict_var.pop('All')
    res_woe = data[var].apply(lambda i: get_category_num(i,dict_var) if not isinstance(i,str) else dict_var['missing'])
    return res_woe

def get_woe_class(path,var,data):
    # 用于预测集数值型变量woe转换
    with open(os.path.join(path,'pkl/%s_woedict.pkl' % var), 'rb') as f: dict_var = pickle.load(f)
    if not os.path.exists(os.path.join(path,'classdict/%s_classdict.pkl'%var)):
        dict_var.pop('All')
        res_woe = data[var].apply(lambda i: get_category_class(i,dict_var))
    else:
        with open(os.path.join(path,'classdict/%s_classdict.pkl' % var), 'rb') as f: dict_class = pickle.load(f)
        res_woe = data[var].apply(lambda i: get_category_class(dict_class[i], dict_var) if i in dict_class.keys() else np.nan)
    return res_woe
def readBinData(binPath):
    with open(os.path.join(binPath, 'pkl/mergeData.pkl'), 'rb') as f:
        cleanData = pickle.load(f)  # 读取清洗好的数据
    with open(os.path.join(binPath, 'pkl/target.pkl'), 'rb') as f:
        train_target = pickle.load(f)  # 读取训练标签
    with open(os.path.join(binPath, 'pkl/var2woe.pkl'), 'rb') as f:
        var2woe = pickle.load(f)  # 读取训练标签
    with open(os.path.join(binPath, 'pkl/iv_get.pkl'), 'rb') as f:
        iv_get = pickle.load(f)  # 读取训练标签

    return cleanData,train_target,var2woe,iv_get


