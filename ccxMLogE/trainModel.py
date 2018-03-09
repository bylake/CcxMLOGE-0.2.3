"""
训练模型
0.demo
1.speed    --> 小网格  大惩罚
2.accuracy --> 超大的网格寻优
3.stable --> 递归 直到Auc满足条件
"""

from ccxmodel.modelmain import ModelMain
import os
import pandas as pd
from ccxmodel.modelutil import ModelUtil
import numpy as np

from ccxMLogE.logModel import ABS_log


def f_genmodelCodeDict(userPath):
    '''
    依据用户输入的路径 生成模型的码表
    :param userPath:
    :return: {k:v} k 模型简称 ccxboost_speed  v 配置文件的绝对路径
    '''
    path = os.path.join(userPath, 'conf')
    filenames = os.listdir(path)
    k = [i.split('.')[0] for i in filenames]
    v = [os.path.join(path, i) for i in filenames]
    dd = {}
    for k_, v_ in zip(k, v):
        dd[k_] = v_
    return dd


def f_xgboost(modelmain, modeltype, modelCode,optimizationType):
    '''

    :param modelmain: 初始化的模型主函数类
    :param modeltype: 模型编码 可取值为 ccxboost + [空,_bayes] + _ + [demo,speed,accuracy,stable]
    :param modelCode: 字典表 模型简称：配置文件绝对路径
    :return: 模型最终计算结果的文件路径
    '''
    if modeltype == 'ccxboost_demo' or modeltype == 'ccxboost_bayes_demo':
        # 演示
        return modelmain.ccxboost_main(modelCode[modeltype],optimizationType)
    elif modeltype == 'ccxboost_speed' or modeltype == 'ccxboost_bayes_speed':
        # 高速
        return modelmain.ccxboost_main(modelCode[modeltype],optimizationType)
    elif modeltype == 'ccxboost_accuracy' or modeltype == 'ccxboost_bayes_accuracy':
        # 精确
        return modelmain.ccxboost_main(modelCode[modeltype],optimizationType)
    elif modeltype == 'ccxboost_stable' or modeltype == 'ccxboost_bayes_stable':
        # 稳定
        # 需封装一个递归的函数 用于跑模型
        train_path, test_path, index_name, target_name = modelmain.train_path, modelmain.test_path, modelmain.index_name, modelmain.target_name
        return f_recursionboostModel(train_path, test_path, index_name, target_name, modelCode[modeltype], 0, optimizationType)


def f_gbm(modelmain, modeltype, modelCode,optimizationType):
    '''
    :param modelmain: 初始化的模型主函数类
    :param modeltype: 模型编码 可取值为 ccxgbm + [空,_bayes] + _ + [demo,speed,accuracy,stable]
    :param modelCode: 字典表 模型简称：配置文件绝对路径
    :return: 模型最终计算结果的文件路径
    '''
    if modeltype == 'ccxgbm_demo' or modeltype == 'ccxgbm_bayes_demo':
        # 演示
        return modelmain.ccxgbm_main(modelCode[modeltype],optimizationType)
    elif modeltype == 'ccxgbm_speed' or modeltype == 'ccxgbm_bayes_speed':
        # 高速
        return modelmain.ccxgbm_main(modelCode[modeltype],optimizationType)
    elif modeltype == 'ccxgbm_accuracy' or modeltype == 'ccxgbm_bayes_accuracy':
        # 精确
        return modelmain.ccxgbm_main(modelCode[modeltype],optimizationType)
    elif modeltype == 'ccxgbm_stable' or modeltype == 'ccxgbm_bayes_stable':
        # 稳定
        # 需封装一个递归的函数 用于跑模型
        train_path, test_path, index_name, target_name = modelmain.train_path, modelmain.test_path, modelmain.index_name, modelmain.target_name
        return f_recursiongbmModel(train_path, test_path, index_name, target_name, modelCode[modeltype], 0, optimizationType)


def f_rf(modelmain, modeltype, modelCode, optimizationType):
    if modeltype == 'ccxrf_demo' or modeltype == 'ccxrf_bayes_demo':
        # 演示
        return modelmain.ccxrf_main(modelCode[modeltype],optimizationType)
    elif modeltype == 'ccxrf_speed' or modeltype == 'ccxrf_bayes_speed':
        # 高速
        return modelmain.ccxrf_main(modelCode[modeltype],optimizationType)
    elif modeltype == 'ccxrf_accuracy' or modeltype == 'ccxrf_bayes_accuracy':
        # 精确
        return modelmain.ccxrf_main(modelCode[modeltype],optimizationType)
    elif modeltype == 'ccxrf_stable' or modeltype == 'ccxrf_bayes_stable':
        # 稳定
        # 需封装一个递归的函数 用于跑模型
        train_path, test_path, index_name, target_name = modelmain.train_path, modelmain.test_path, modelmain.index_name, modelmain.target_name
        return f_recursionrfModel(train_path, test_path, index_name, target_name, modelCode[modeltype], 0, optimizationType)


#手动寻参的文件生成
def create_conf(arithmetic,param,userPath):
    path = os.path.join(userPath, 'conf')   #path为配置文件的文件夹地址
    import time
    if arithmetic == 'Xgboost':
        path_xgboost = os.path.join(path,'xgboostManual'+str(int(time.time() * 1000))+'.conf')
        with open(path_xgboost,'w') as f:
            file1 = r'''[DIRECTORY]
project_pt =%s
''' %os.path.join(userPath, 'ccxboost/')
            file1 = file1 + r'''data_pt = %(project_pt)s/data/
log_pt = %(project_pt)s/modellog/
mid_result = %(project_pt)s/modeldata/
fig_pt = %(project_pt)s/modelfig/
model_re_pt = %(project_pt)s/modeltxt/

[XGB_PARAMS]
objective = binary:logistic
eval_metric = auc
verbose_eval = 3
'''
            #分段的原因为加上%s字符会出错
            file1 = file1 + r'''eta = %s
max_depth = %s
subsample = %s
colsample_bytree = %s
min_child_weight = %s
gamma = %s
lambda = %s
scale_pos_weight = %s

[XGB_OPTIONS]
cv_mess = ccxboost
num_round = %s
num_iter = %s
init_points = %s
silent = 1
random_state = 701
only_test = %s
'''%(param['eta'],param['max_depth'],
     param['subsample'],param['colsample_bytree'],
     param['min_child_weight'],param['gamma'],
     param['lambda'],param['scale_pos_weight'],
     param['num_round'],param['num_iter'],
     param['init_points'],param['only_test'])
            #字符串组装完成
            f.write(file1)
        return path_xgboost
    elif arithmetic == 'GBM':
        path_gbm = os.path.join(path,'GBMManual'+str(int(time.time() * 1000))+'.conf')
        with open(path_gbm,'w') as f:
            file1 = r'''[DIRECTORY]
project_pt =%s
''' %os.path.join(userPath, 'ccxgbm/')
            file1 = file1 + r'''data_pt = %(project_pt)s/data/
log_pt = %(project_pt)s/modellog/
mid_result = %(project_pt)s/modeldata/
fig_pt = %(project_pt)s/modelfig/
model_re_pt = %(project_pt)s/modeltxt/

[CROSS_VALIDATION]
cv_num = 5

[GBM_PARAMS]
objective = binary
metric = auc
'''
            #分段的原因为加上%s字符会出错
            file1 = file1 + r'''boosting_type = %s
colsample_bytree =%s
learning_rate = %s
min_child_weight = %s
min_gain_to_split = %s
num_leaves = %s
reg_lambda = %s
subsample = %s
is_unbalance = %s

[GBM_OPTIONS]
num_round = %s
cv_mess = ccxgbm
num_iter = %s
init_points = %s
only_test = %s
'''%(param['boosting_type'],param['colsample_bytree'],param['learning_rate'],
     param['min_child_weight'],param['min_gain_to_split'],
     param['num_leaves'],param['reg_lambda'],
     param['subsample'],param['is_unbalance'],
     param['num_round'],param['num_iter'],
     param['init_points'],param['only_test'])
            #字符串组装完成
            f.write(file1)
        return path_gbm
    elif arithmetic == 'RF':
        path_RF = os.path.join(path,'RFManual'+str(int(time.time() * 1000))+'.conf')
        with open(path_RF,'w') as f:
            file1 = r'''[DIRECTORY]
project_pt =%s
''' %os.path.join(userPath, 'ccxrf/')
            file1 = file1 + r'''data_pt = %(project_pt)s/data/
log_pt = %(project_pt)s/modellog/
mid_result = %(project_pt)s/modeldata/
fig_pt = %(project_pt)s/modelfig/
model_re_pt = %(project_pt)s/modeltxt/

[RF_PARAMS]
'''
            #分段的原因为加上%s字符会出错
            file1 = file1 + r'''max_depth = %s
max_features = %s
oob_score = True
n_estimators = %s
min_samples_split = %s
min_samples_leaf = %s
criterion = %s

[RF_OPTIONS]
init_points = %s
num_iter = %s
cv_mess = ccxrf
only_test = %s
'''%(param['max_depth'],param['max_features'],
     param['n_estimators'],param['min_samples_split'],
     param['min_samples_leaf'],param['criterion'],
     param['init_points'],param['num_iter'],param['only_test'])
            #字符串组装完成
            f.write(file1)
        return path_RF
    else:
        print('arithmetic为'+str(arithmetic)+",格式有问题")
        return ''


@ABS_log('MLogEDebug')
def f_trainModelMain(train_path, test_path, index_name, target_name, userPath, modeltype, arithmetic,optimizationType,is_auto,param):
    '''
    调用模型的主函数
    :param train_path: 训练集 str DataFrame
    :param test_path: 测试集 str DataFrame
    :param index_name: 索引
    :param target_name: 目标变量列名
    :param userPath: 用户路径
    :param modeltype: 模型的简称 有24个取值 [grid,bayes] + [ccxboost,ccxgbm,ccxrf] + [demo,speed,accuracy,stable]
    :param arithmetic: [Xgboost,GBM,RF]
    :param optimizationType: [grid,bayes]
    :param is_auto: 是否自动寻参
    :param param: 自动寻参的超参数范围
    :return: 最优模型结果的文件路径
    '''
    modelCode = f_genmodelCodeDict(userPath)
    modelmain = ModelMain(train_path, test_path, index_name, target_name)

    #自动寻优的入口 is_auto==0 表示手动  仅仅bayes方式存在手动调参
    if int(is_auto) == 0 and optimizationType == 'bayes':
        try:
            #进来了  下面就是自动寻参的天下了
            manual_conf_path = create_conf(arithmetic, param,userPath)
            if arithmetic == 'Xgboost':
                return f_recursionboostModel(train_path, test_path, index_name, target_name, manual_conf_path, 0,
                                      optimizationType)
            elif arithmetic == 'GBM':
                return f_recursiongbmModel(train_path, test_path, index_name, target_name, manual_conf_path, 0,
                                      optimizationType)
            elif arithmetic == 'RF':
                return f_recursionrfModel(train_path, test_path, index_name, target_name, manual_conf_path, 0,
                                      optimizationType)
            else:
                # 写日志
                print('错误码005 模型没有跑起来')
                return []

        except Exception as e:
            print('错误码004 手动寻参出错问题为：')
            print("ErrorMsg:\t"+str(e))
            return []

    if arithmetic == 'Xgboost':
        return f_xgboost(modelmain, modeltype, modelCode,optimizationType)
    elif arithmetic == 'GBM':
        return f_gbm(modelmain, modeltype, modelCode,optimizationType)
    elif arithmetic == 'RF':
        return f_rf(modelmain, modeltype, modelCode,optimizationType)
    else:
        # 写日志
        print('错误码003 模型没有跑起来')
        return []




# 递归调用的停止条件：
# 1.上一次入模变量个数 == 重要变量个数 即已经无法剔除变量
# 2.Auc train_auc - test_auc < 0.015 即可停止 （需考虑 如果变量太少，一致达不到这个要求怎么办）


def f_recursionboostModel(train_path, test_path, index_name, target_name, modelconf, i,optimizationType):
    '''
    递归的将上一轮的重要变量重新作为输入 从而达到筛选变量的作用
    :param train_path: 训练集
    :param test_path: 测试集
    :param index_name:
    :param target_name:
    :param modelconf: 模型配置文件路径
    :param i: 记录递归的次数
    :return: 最终递归完成的模型输出 结果的路径列表
    '''
    train_path = ModelUtil.load_data(train_path)
    test_path = ModelUtil.load_data(test_path)
    initmodelmain = ModelMain(train_path, test_path, index_name, target_name)
    initpathlist = initmodelmain.ccxboost_main(modelconf,optimizationType)

    # 1.计算出重要变量的个数
    implen, impvar = f_getImplen(initpathlist[2])
    # 2.计算出模型的AUC和KS
    train_auc, train_ks = f_getAucKs(initpathlist[3])
    test_auc, test_ks = f_getAucKs(initpathlist[4])
    # 3.判断出模型重要变量占总变量的百分比情况
    imppct = f_getVarpctboost(initpathlist[1], implen)  # 入模变量 == 重要变量
    flag = f_flag(train_auc, train_ks, test_auc, test_ks, imppct)
    i = i + 1
    if i < 5:
        if flag:
            print('递归调用 ' * 20)
            newselectcol = impvar + [index_name, target_name]
            print('---入选模型的变量个数%s' % len(newselectcol))
            train_path = ModelUtil.load_data(train_path)[newselectcol]
            test_path = ModelUtil.load_data(test_path)[newselectcol]
            print('##' * 20, i, '##' * 20)
            # 后续优化 递归的同时修改配置文件modelconf
            return f_recursionboostModel(train_path, test_path, index_name, target_name, modelconf, i,optimizationType)

        else:
            print('满足条件结束递归 ' * 10)
            return initpathlist
    else:
        print('递归次数达到要求结束递归' * 10)
        return initpathlist


def f_recursiongbmModel(train_path, test_path, index_name, target_name, modelconf, i,optimizationType):
    '''
    递归的将上一轮的重要变量重新作为输入 从而达到筛选变量的作用
    :param train_path: 训练集
    :param test_path: 测试集
    :param index_name:
    :param target_name:
    :param modelconf: 模型配置文件路径
    :param i: 记录递归的次数
    :return: 最终递归完成的模型输出 结果的路径列表
    '''
    train_path = ModelUtil.load_data(train_path)
    test_path = ModelUtil.load_data(test_path)
    initmodelmain = ModelMain(train_path, test_path, index_name, target_name)
    initpathlist = initmodelmain.ccxgbm_main(modelconf,optimizationType)

    # 1.计算出重要变量的个数
    implen, impvar = f_getImplen(initpathlist[2])
    # 2.计算出模型的AUC和KS
    train_auc, train_ks = f_getAucKs(initpathlist[3])
    test_auc, test_ks = f_getAucKs(initpathlist[4])
    # 3.判断出模型重要变量占总变量的百分比情况
    imppct = f_getVarpctgbm(initpathlist[1], implen)  # 入模变量 == 重要变量
    flag = f_flag(train_auc, train_ks, test_auc, test_ks, imppct)
    i = i + 1
    if i < 5:
        if flag:
            print('递归调用 ' * 20)
            newselectcol = impvar + [index_name, target_name]
            print('---入选模型的变量个数%s' % len(newselectcol))
            train_path = ModelUtil.load_data(train_path)[newselectcol]
            test_path = ModelUtil.load_data(test_path)[newselectcol]
            print('##' * 20, i, '##' * 20)
            # 后续优化 递归的同时修改配置文件modelconf
            return f_recursiongbmModel(train_path, test_path, index_name, target_name, modelconf, i,optimizationType)

        else:
            print('满足条件结束递归 ' * 10)
            return initpathlist
    else:
        print('递归次数达到要求结束递归' * 10)
        return initpathlist


def f_recursionrfModel(train_path, test_path, index_name, target_name, modelconf, i,optimizationType):
    '''
    递归的将上一轮的重要变量重新作为输入 从而达到筛选变量的作用
    :param train_path: 训练集
    :param test_path: 测试集
    :param index_name:
    :param target_name:
    :param modelconf: 模型配置文件路径
    :param i: 记录递归的次数
    :return: 最终递归完成的模型输出 结果的路径列表
    '''
    train_path = ModelUtil.load_data(train_path)
    test_path = ModelUtil.load_data(test_path)
    initmodelmain = ModelMain(train_path, test_path, index_name, target_name)
    initpathlist = initmodelmain.ccxrf_main(modelconf,optimizationType)

    # 1.计算出重要变量的个数
    implen, impvar = f_getImplen(initpathlist[2])
    # 2.计算出模型的AUC和KS
    train_auc, train_ks = f_getAucKs(initpathlist[3])
    test_auc, test_ks = f_getAucKs(initpathlist[4])
    # 3.判断出模型重要变量占总变量的百分比情况
    imppct = f_getVarpctrf(initpathlist[1], implen)  # 入模变量 == 重要变量
    flag = f_flag(train_auc, train_ks, test_auc, test_ks, imppct)
    i = i + 1
    if i < 5:
        if flag:
            print('递归调用 ' * 20)
            newselectcol = impvar + [index_name, target_name]
            print('---入选模型的变量个数%s' % len(newselectcol))
            train_path = ModelUtil.load_data(train_path)[newselectcol]
            test_path = ModelUtil.load_data(test_path)[newselectcol]
            print('##' * 20, i, '##' * 20)
            # 后续优化 递归的同时修改配置文件modelconf
            return f_recursionrfModel(train_path, test_path, index_name, target_name, modelconf, i,optimizationType)

        else:
            print('满足条件结束递归 ' * 10)
            return initpathlist
    else:
        print('递归次数达到要求结束递归' * 10)
        return initpathlist


def f_flag(train_auc, train_ks, test_auc, test_ks, imppct):
    '''
    判断是否递归的条件
    :param train_auc:
    :param train_ks:
    :param test_auc:
    :param test_ks:
    :param imppct: 入模变量和重要变量个数是否相等
    :return:
    '''
    aucgap = np.round(train_auc - test_auc, 3)
    ksgap = np.round(train_ks - test_ks, 2)

    if aucgap < 0.015 or ksgap < 0.03 or imppct:
        return False  # 不再递归
    else:
        return True  # 递归


def f_getImplen(imp_path):
    '''
    得到重要变量的长度 和具体的重要变量
    :param imp_path:
    :return:
    '''
    dd = pd.read_csv(imp_path)
    col = dd.Feature_Name.values.tolist()
    return len(dd), col  # 可能有编码问题 尤其是变量名有汉字的情况下


def f_getAucKs(pretrpath):
    '''
    计算出AUC和 KS
    :param pretrpath: 使用最优模型预测出的结果路径
    :return:
    '''
    data = pd.read_csv(pretrpath)  # 读入数据 index_name ,target_name, P_value
    ks = ModelUtil.ks(data.P_value, data.iloc[:, 1])
    Auc = ModelUtil.AUC(data.P_value, data.iloc[:, 1])
    return Auc, ks


def f_getVarpctboost(model_path, implen):
    '''
    依据模型路径 给出需要输入模型的变量个数
    :param model_path: 模型路径
    :param implen: 重要变量长度
    :return:
    '''
    x = ModelUtil.load_bstmodel(model_path)
    try:
        # xgboost 获取变量的方法
        x = x.feature_names
        modellen = len(x)
    except:
        modellen = np.nan

    return implen == modellen


def f_getVarpctgbm(model_path, implen):
    '''
    依据模型路径 给出需要输入模型的变量个数
    :param model_path: 模型路径
    :param implen: 重要变量长度
    :return:
    '''
    x = ModelUtil.load_bstmodel(model_path)
    try:
        # gbm 获取入模变量的方法
        modellen = len(x.feature_name())
    except:
        modellen = np.nan

    return implen == modellen


def f_getVarpctrf(model_path, implen):
    '''
    依据模型路径 给出需要输入模型的变量个数
    :param model_path: 模型路径
    :param implen: 重要变量长度
    :return:
    '''
    x = ModelUtil.load_bstmodel(model_path)
    try:
        # 随机森林的获取方法
        modellen = x.n_features_
    except:
        modellen = np.nan

    return implen == modellen


if __name__ == '__main__':
    # userPath = r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit'
    # train_path = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\train_base14.csv'
    # test_path = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\test_base14.csv'
    # index_name = 'contract_id'
    # target_name = 'target'
    # modeltype, arithmetic = 'ccxboost_demo', 'Xgboost'
    # f_trainModel(train_path, test_path, index_name, target_name, userPath, modeltype, arithmetic)
    #
    # modeltype, arithmetic = 'ccxgbm_speed', 'GBM'
    # f_trainModel(train_path, test_path, index_name, target_name, userPath, modeltype, arithmetic)
    #
    # modeltype, arithmetic = 'ccxrf_accuracy', 'RF'
    # f_trainModel(train_path, test_path, index_name, target_name, userPath, modeltype, arithmetic)
    #
    # pretrpath = r'C:\Users\liyin\Desktop\ccxmodel\TestUnit\ccxgbm\model20171129120528\modeldata\d_2017-11-29_train_predict.csv'
    # f_getAucKs(pretrpath)

    ###测试递归调用的情况

    # train_path = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\train_base14.csv'
    # test_path = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\test_base14.csv'
    # index_name = 'contract_id'
    # target_name = 'target'
    # modelconf = r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\conf\ccxboost_demo.conf'
    # f_recursionModel(train_path, test_path, index_name, target_name, modelconf, 0)
    #
    # modelconf = r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\conf\ccxgbm_speed.conf'
    # f_recursionModel(train_path, test_path, index_name, target_name, modelconf, 0)
    #
    # modelconf = r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\conf\ccxrf_speed.conf'
    # f_recursionModel(train_path, test_path, index_name, target_name, modelconf, 0)

    # 测试发现存在的功能的缺陷
    # 1. 由于初始参数造成的，超参数是的迭代不能再筛选变量 但还是不能满足ks gap的要求

    # 最终的调用测试
    userPath = r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit'
    train_path = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\train_base14.csv'
    test_path = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\test_base14.csv'
    index_name = 'contract_id'
    target_name = 'target'
    modeltype, arithmetic = 'ccxboost_stable', 'Xgboost'
    f_trainModelMain(train_path, test_path, index_name, target_name, userPath, modeltype, arithmetic)

    modeltype, arithmetic = 'ccxgbm_stable', 'GBM'
    f_trainModelMain(train_path, test_path, index_name, target_name, userPath, modeltype, arithmetic)

    modeltype, arithmetic = 'ccxrf_stable', 'RF'
    f_trainModelMain(train_path, test_path, index_name, target_name, userPath, modeltype, arithmetic)

    train_path = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\train_base14.csv'
    data = pd.read_csv(train_path)
    data = data.drop_duplicates('contract_id')

    data.head(5000).to_csv(r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\train_base14_1.csv', index=False)
