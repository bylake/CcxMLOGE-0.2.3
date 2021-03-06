"""
输出的json格式 定义 转换 等
"""
import pandas as pd
import os
from datetime import datetime
import simplejson
import pandas_profiling
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
# type 1 接口的输出
# part2 = {
#     "datasetInfo": {"数据集名称": "data_demo", "样本量": 20000, "维度": 200},
#     "varSummary": {"cateVar": [{'IV': 0.01,
#                                 'Type': 'category',
#                                 'missingN': 0,
#                                 'missing_pct': 0,
#                                 'nunique': 2,
#                                 'top1': 1,
#                                 'top1_pct': 95.24,
#                                 'top2': 0,
#                                 'top2_pct': 4.76,
#                                 'top3': None,
#                                 'top3_pct': None,
#                                 'vList': '[0,1]',
#                                 'varName': 'regprov_isequal_liveprov'},
#                                {'IV': 0.01,
#                                 'Type': 'category',
#                                 'missingN': 0,
#                                 'missing_pct': 0,
#                                 'nunique': 2,
#                                 'top1': 1,
#                                 'top1_pct': 95.24,
#                                 'top2': 0,
#                                 'top2_pct': 4.76,
#                                 'top3': None,
#                                 'top3_pct': None,
#                                 'vList': '[0,1]',
#                                 'varName': 'regprov_isequal_liveprov'}
#
#                                ],
#                    "numVar": [{'IV': 0.01,
#                                'Type': 'numric',
#                                'max': 172179.8,
#                                'mean': 37562.1559456181,
#                                'median': 32294.8891325,
#                                'min': 590,
#                                'missingN': 100334,
#                                'missing_pct': 99.33,
#                                'quantile1': 15143.94023275,
#                                'quantile3': 50249.75,
#                                'range': '[590,172180]',
#                                'std': 30375.4095940718,
#                                'varName': 'overdue_cuiqian_meanAmount'}
#                        , {'IV': 0.01,
#                           'Type': 'numric',
#                           'max': 172179.8,
#                           'mean': 37562.1559456181,
#                           'median': 32294.8891325,
#                           'min': 590,
#                           'missingN': 100334,
#                           'missing_pct': 99.33,
#                           'quantile1': 15143.94023275,
#                           'quantile3': 50249.75,
#                           'range': '[590,172180]',
#                           'std': 30375.4095940718,
#                           'varName': 'overdue_cuiqian_meanAmount'}
#
#                               ]},
#     "detailVarPath": {"path": "ccc/ccc/sdcfef"}
#  }
# Type1 = {
#     "reqId": "请求ID",
#     "reqTime": "请求时间戳",
#     "type": 1,
#     "dataDescription": part2,
#     "variableAnalysis": None,
#     "modelOutput": None,
#     "otherOutput": None
#
# }
from ccxmodel.modelutil import ModelUtil
from sklearn.metrics import roc_curve
from ccxMLogE.logModel import ABS_log
from ccxMLogE.trainModel import f_getAucKs
from ccxMLogE.varDescSummary import f_rawbins, IV, f_VardescWriter


def f_detailVarhtml(df, userPath):
    '''
    详细变量的输出情况
    :param df:
    :return:
    '''
    reqTime = datetime.today().strftime('%Y-%m-%d_%H%M%S')
    respath = f_mkdir(userPath, 'vardesc')
    profile = pandas_profiling.ProfileReport(df)
    filename = 'detailVarhtml_' + reqTime + '.html'
    path = os.path.join(respath, filename)
    profile.to_file(outputfile=path)
    return path


@ABS_log('MLogEDebug')
def f_part2Output(resdesc, userPath, df):
    '''
    数据集的描述性分析结果
    :param resdesc: 元组 numVardesc, cateVardesc, detailVarIV, dd, one_hot, cate_2, dropcol
    :param userPath 用户路径
    :return:
    '''
    # numVardesc, cateVardesc 转字典
    # detailVarIV 写入csv中，返回路径
    if resdesc:  # 过滤None
        numVardesc = resdesc[0]
        cateVardesc = resdesc[1]
        detailVarIV = resdesc[2]
    else:
        numVardesc = pd.DataFrame()
        cateVardesc = pd.DataFrame()
        detailVarIV = pd.DataFrame()

    numVar = numVardesc.to_dict(orient='records')
    cateVar = cateVardesc.to_dict(orient='records')

    # 第二部分这里的是返回html文件路径
    # 发现 同步异步都要计算这个的话 会很麻烦
    path_ = f_detailVarhtml(df, userPath)

    # 这是第三部分的事情了 1204 新增了文件的时间戳
    creattime = datetime.today().strftime('%Y-%m-%d_%H%M%S')
    filename = 'detailVarIV' + '_' + creattime + '.csv'
    # 1211 新增的 存储文件夹
    respath = f_mkdir(userPath, 'vardesc')
    path = os.path.join(respath, filename)
    if len(detailVarIV) > 1:
        detailVarIV.to_csv(path, index=False)
    else:
        path = None

    # 1206 发现了int64 不能被序列化的bug 这个bug的出现位置就是这个地方

    return {"varSummary": {"cateVar": cateVar, "numVar": numVar}, "detailVarPath": {"path": path_}}, path


class MyEncoder(simplejson.JSONEncoder):
    '''
    此自定义类为了解决 int 不能被序列化的bug
    '''

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


@ABS_log('MLogEDebug')
def f_part2Output4yibu(resdesc, userPath):
    '''
    数据集的描述性分析结果 为了异步服务 异步时 不会计算html 页面
    :param resdesc: 元组 numVardesc, cateVardesc, detailVarIV, dd, one_hot, cate_2, dropcol
    :param userPath 用户路径
    :return:
    '''
    # numVardesc, cateVardesc 转字典
    # detailVarIV 写入csv中，返回路径
    if resdesc:  # 过滤None
        numVardesc = resdesc[0]
        cateVardesc = resdesc[1]
        detailVarIV = resdesc[2]
    else:
        numVardesc = pd.DataFrame()  # 1211 修改 要不然后续的.to_dict(orient='records')会报错
        cateVardesc = pd.DataFrame()
        detailVarIV = pd.DataFrame()

    numVar = numVardesc.to_dict(orient='records')
    cateVar = cateVardesc.to_dict(orient='records')

    # 第二部分这里的是返回html文件路径
    # 发现 同步异步都要计算这个的话 会很麻烦 异步将不会计算了
    # path_ = f_detailVarhtml(df, userPath)

    # 这是第三部分的事情了
    # 这是第三部分的事情了 1204 新增了文件的时间戳
    creattime = datetime.today().strftime('%Y-%m-%d_%H%M%S')
    filename = 'detailVarIV' + '_' + creattime + '.csv'
    # 1211 新增的 存储文件夹
    respath = f_mkdir(userPath, 'vardesc')
    path = os.path.join(respath, filename)
    if len(detailVarIV) > 1:
        detailVarIV.to_csv(path, index=False)
    else:
        path = None

    return {"varSummary": {"cateVar": cateVar, "numVar": numVar}, "detailVarPath": {"path": None}}, path


@ABS_log('MLogEDebug')
def f_type1Output(reqId, datasetInfo, descout, path):
    part2 = dict({"datasetInfo": datasetInfo}, **descout)
    part3_1 = {"impVar": None, "topNpath": path}
    reqTime = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    dict1 = {"reqId": reqId, "reqTime": reqTime, "type": 1,
             "dataDescription": part2,
             "variableAnalysis": part3_1,
             "modelOutput": None,
             "otherOutput": None
             }
    return simplejson.dumps(dict1, ensure_ascii=False, ignore_nan=True, cls=MyEncoder)


##

# part3 = {
#     "impVar": [{'Feature_Name': 'asset_grad_E',
#                 'gain': 16.8522672727272,
#                 'pct_importance': 0.045},
#                {'Feature_Name': 'asset_grad_D', 'gain': 14.697476, 'pct_importance': 0.0393},
#                {'Feature_Name': '_age_5.0',
#                 'gain': 11.5207033333333,
#                 'pct_importance': 0.0308}]
#     ,
#     "topNpath": "xxx/xxx/xxx.csv"
# }
#

def f_part3Output(imppath, topNpath, dfcolnames):
    '''
    变量重要性分析
    :param imppath: 重要变量结果路径
    :param topNpath: 第二部分提前计算好的所有变量的IV详情
    :param dfcolnames 原始数据的字段
    :return: 结果字典 {}
    '''
    impVar = f_readdata(imppath)
    # topN这里可以优化 往回追到one-hot之前的变量列表 在只显示全部的重要变量
    # 开发一个函数 依据one-hot后的变量 找到没有one-hot之前的变量名
    topNames = f_getRawcolnames(impVar.Feature_Name, dfcolnames)
    topNames.sort_values(by=['rank'], inplace=True)
    AllIv = f_readdata(topNpath)
    topNIV = pd.merge(topNames, AllIv, left_on='varname', right_on='varname', how='left')
    # 1204 新增 增加了时间戳 并发请求时避免线程不安全性
    creattime = datetime.today().strftime('%Y-%m-%d_%H%M%S')
    filename = 'TopNVarIV' + '_' + creattime + '.csv'
    topNpath = os.path.join(os.path.dirname(topNpath), filename)
    topNIV.to_csv(topNpath, index=False)
    print('重要变量IV值已经保存成功： ', topNpath)

    return {"impVar": impVar.to_dict(orient='records'), "topNpath": topNpath}


def f_readdata(path):
    '''
    1208发现的bug
    读取数据 由于原始数据中有中文 导致了 我存储IV时 编码为gbk 使用pd.read_csv 时 默认编码为utf-8
    :param path:
    :return:
    '''
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='gbk')
    except:
        print('--读取全部数据的IV结果出错--')


def f_getRawcolnames(impVarcol, dfcolnames):
    '''
    依据模型选出的重要变量 找到原始的变量名
    :param impVarcol: 列表
    :param dfcolnames: 列表
    :return:
    '''
    ls = []
    rankls = []
    for i in dfcolnames:
        # 拿到原始的字段名
        for j, rank in zip(impVarcol, np.arange(len(impVarcol)) + 1):
            # 12月11日 有个bug, bug原因 V28 in V282_1.0 ,先找到V28 后找到V282
            # i 短 j 长
            if f_find(str(i), str(j)):
                ls.append(i)
                rankls.append(rank)
                break

    return pd.DataFrame({'varname': ls, 'rank': rankls})


def f_find(rawcol, onehotcol):
    '''
    1211日发现bug，原始变量名短 onehotcol变量名长 V28 in V282_1.0 找到了V28而不是V282
    :param rawcol: 原始字段列名
    :param onehotcol: onehot后的变量列名
    :return:
    后期优化思路：ont-hot时就生成一个对应的字典 是最稳妥的
    '''
    x = onehotcol.split('_')  # 用下划线切一下
    if len(x) == 1:
        # 说明这个onehot的变量没有下划线，那就说明了原始变量没有经过onehot处理
        return rawcol == onehotcol
    else:
        # 说明了onehot中有下划线，下划线的来源有两个，一个是原始变量中有，一个是原始变量中没有
        if '_' in rawcol:
            # 原始变量中有下划线
            y = len(rawcol.split('_'))
            if len(x) == y:
                # 说明了onehot中的下划线来源于原始变量中下划线
                return rawcol == onehotcol
            elif len(x) == y + 1:
                # 说明了下划线来源于one_hot
                return rawcol == '_'.join(x[0:-1])
        else:
            # 原始变量中没有下划线 则下划线来源于onehot之后
            return rawcol == '_'.join(x[0:-1])


# 测一下topN
# imppath = r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\ccxboost\model20171201200539\modeldata\d_2017-12-01_importance_var.csv'
# topNpath = r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\detailVarIV.csv'
# dfcolnames = pd.read_csv(r'C:\\Users\\liyin\\Desktop\\20170620_tn\\0620_base\\train_base14.csv').columns
#
# zz = f_part3Output(imppath, topNpath, dfcolnames)


# part4 = {
#     "modeldataInfo": [{'入模维度': 61.0,
#                        '正负样本比': 5.9500000000000002,
#                        '总维度': 3145.0,
#                        '样本量': 9661.0,
#                        '正样本比例': 0.14380000000000001,
#                        '重要变量': 58.0},
#                       {'入模维度': 61.0,
#                        '正负样本比': 5.9500000000000002,
#                        '总维度': 3145.0,
#                        '样本量': 4141.0,
#                        '正样本比例': 0.14380000000000001,
#                        '重要变量': 58.0}],
#
#     "modelreport": [{'AUC': 0.69367500000000004,
#                      'KS': 0.29683399999999999,
#                      'f1-score': 0.80000000000000004,
#                      'gini': 0.38735000000000008,
#                      'precision': 0.84999999999999998,
#                      'recall': 0.85999999999999999,
#                      'support': 14422.0},
#                     {'AUC': 0.64169900000000002,
#                      'KS': 0.221996,
#                      'f1-score': 0.79000000000000004,
#                      'gini': 0.28339800000000004,
#                      'precision': 0.82999999999999996,
#                      'recall': 0.85999999999999999,
#                      'support': 6182.0}]
#     ,
#     "aucksPlot": {'trainKSpath': 'xxxx.csv',
#                   'trainAUCpath': 'xxxx.csv',
#                   'testKSpath': 'xxxx.csv',
#                   'testAUCpath': 'xxxx.csv',
#                   }
#     ,
#     "pvalueReport": [{'IV': 0.412686951544129,
#                       'bad': 6,
#                       'bad_per': 0.75,
#                       'bins_score': '(590, 600]',
#                       'factor_per': 0.000579626141138965,
#                       'good': 2,
#                       'model_pvalue': 0.527831077575683,
#                       'total': 8},
#                      {'IV': 0.412686951544129,
#                       'bad': 20,
#                       'bad_per': 0.571428571428571,
#                       'bins_score': '(600, 610]',
#                       'factor_per': 0.00253586436748297,
#                       'good': 15,
#                       'model_pvalue': 0.442838847637176,
#                       'total': 35},
#                      {'IV': 0.412686951544129,
#                       'bad': 33,
#                       'bad_per': 0.407407407407407,
#                       'bins_score': '(610, 620]',
#                       'factor_per': 0.00586871467903202,
#                       'good': 48,
#                       'model_pvalue': 0.362247616052627,
#                       'total': 81},
#                      {'IV': 0.412686951544129,
#                       'bad': 147,
#                       'bad_per': 0.402739726027397,
#                       'bins_score': '(620, 630]',
#                       'factor_per': 0.0264454426894652,
#                       'good': 218,
#                       'model_pvalue': 0.285614490509033,
#                       'total': 365},
#                      {'IV': 0.412686951544129,
#                       'bad': 329,
#                       'bad_per': 0.285590277777777,
#                       'bins_score': '(630, 640]',
#                       'factor_per': 0.083466164324011,
#                       'good': 823,
#                       'model_pvalue': 0.220752000808715,
#                       'total': 1152},
#                      {'IV': 0.412686951544129,
#                       'bad': 713,
#                       'bad_per': 0.183149242229642,
#                       'bins_score': '(640, 650]',
#                       'factor_per': 0.282060570931749,
#                       'good': 3180,
#                       'model_pvalue': 0.168598234653472,
#                       'total': 3893},
#                      {'IV': 0.412686951544129,
#                       'bad': 498,
#                       'bad_per': 0.106798198584602,
#                       'bins_score': '(650, 660]',
#                       'factor_per': 0.337849587016374,
#                       'good': 4165,
#                       'model_pvalue': 0.128025189042091,
#                       'total': 4663},
#                      {'IV': 0.412686951544129,
#                       'bad': 217,
#                       'bad_per': 0.0693512304250559,
#                       'bins_score': '(660, 670]',
#                       'factor_per': 0.226706274452977,
#                       'good': 2912,
#                       'model_pvalue': 0.0971796214580535,
#                       'total': 3129},
#                      {'IV': 0.412686951544129,
#                       'bad': 21,
#                       'bad_per': 0.0443974630021141,
#                       'bins_score': '(670, 680]',
#                       'factor_per': 0.0342703955948413,
#                       'good': 452,
#                       'model_pvalue': 0.0732611268758773,
#                       'total': 473},
#                      {'IV': 0.412686951544129,
#                       'bad': 1,
#                       'bad_per': 0.333333333333333,
#                       'bins_score': '(680, 690]',
#                       'factor_per': 0.000217359802927112,
#                       'good': 2,
#                       'model_pvalue': 0.0569750107824802,
#                       'total': 3},
#                      {'IV': 0.412686951544129,
#                       'bad': 1985,
#                       'bad_per': 0.143819736270105,
#                       'bins_score': 'All',
#                       'factor_per': 1.0,
#                       'good': 11817,
#                       'model_pvalue': None,
#                       'total': 13802}]
#
# }


def f_part4Output(repathlist, train, test, target_name, userPath):
    # modeldataInfo
    modeldataInfo = f_modeldataInfo(repathlist, train, test, target_name)

    # modelreport
    trpredpath = repathlist[3]
    tepredpath = repathlist[4]
    trpred = pd.read_csv(trpredpath)
    tepred = pd.read_csv(tepredpath)

    trp = precision_score(trpred.iloc[:, 1], trpred.P_value > 0.5, average='binary')
    trr = recall_score(trpred.iloc[:, 1], trpred.P_value > 0.5, average='binary')
    trf1score = f1_score(trpred.iloc[:, 1], trpred.P_value > 0.5, average='binary')
    trauc, trks = f_getAucKs(trpredpath)
    trgini = trauc * 2 - 1
    trsupport = trpred.shape[0]

    tep = precision_score(tepred.iloc[:, 1], tepred.P_value > 0.5, average='binary')
    ter = recall_score(tepred.iloc[:, 1], tepred.P_value > 0.5, average='binary')
    tef1score = f1_score(tepred.iloc[:, 1], tepred.P_value > 0.5, average='binary')
    teauc, teks = f_getAucKs(tepredpath)
    tegini = teauc * 2 - 1
    tesupport = tepred.shape[0]

    re = {'AUC': [trauc, teauc],
          'KS': [trks, teks],
          'f1-score': [trf1score, tef1score],
          'gini': [trgini, tegini],
          'precision': [trp, tep],
          'recall': [trr, ter],
          'support': [trsupport, tesupport]}

    modelreport = pd.DataFrame(re).to_dict(orient='records')

    # aucksPlot
    fpr, tpr, thresholds = roc_curve(trpred.iloc[:, 1], trpred.P_value)

    trks = pd.DataFrame([range(len(tpr)), tpr, fpr, tpr - fpr], index=['x', 'tpr', 'fpr', 'ks']).T
    trauc = pd.DataFrame([fpr, tpr], index=['x', 'y']).T

    fpr, tpr, thresholds = roc_curve(tepred.iloc[:, 1], tepred.P_value)
    teks = pd.DataFrame([range(len(tpr)), tpr, fpr, tpr - fpr], index=['x', 'tpr', 'fpr', 'ks']).T
    teauc = pd.DataFrame([fpr, tpr], index=['x', 'y']).T

    # 1211 优化 存储目录更为人性化
    respath = f_mkdir(userPath, 'modelres')

    trkspath = os.path.join(respath, 'trainKSpath.csv')
    traucpath = os.path.join(respath, 'trainAUCpath.csv')
    tekspath = os.path.join(respath, 'testKSpath.csv')
    teaucpath = os.path.join(respath, 'testAUCpath.csv')

    trks.to_csv(trkspath, index=False)
    trauc.to_csv(traucpath, index=False)

    teks.to_csv(tekspath, index=False)
    teauc.to_csv(teaucpath, index=False)

    aucksPlot = {'trainKSpath': trkspath,
                 'trainAUCpath': traucpath,
                 'testKSpath': tekspath,
                 'testAUCpath': teaucpath,
                 }

    # pvalueReport
    # 这次先显示测试集的，后续的将测试集的加上
    # 明天来优化一下概率分箱的函数
    # pvalueReport = f_NumVarIV(tepred.P_value * 100, tepred[target_name])
    # pvalueReport['bins'] = pvalueReport['bins'].apply(str)
    trpvalueReport = f_pvalueReport(trpred)
    trpvalueReport['modelDataName'] = 'train'
    tepvalueReport = f_pvalueReport(tepred)
    tepvalueReport['modelDataName'] = 'test'
    pvalueReport = pd.concat([trpvalueReport, tepvalueReport])
    pvalueReport = pvalueReport.to_dict(orient='records')

    return {"modeldataInfo": modeldataInfo, "modelreport": modelreport,
            "aucksPlot": aucksPlot, "pvalueReport": pvalueReport
            }


def f_zcPvalue(x):
    '''
    离一个数最近的整数 且 能被10, 5等整除整除,且离原始数据更近
    :return:
    '''
    if np.ceil(x) % 10 == 0 or np.ceil(x) % 5 == 0:
        return np.ceil(x)
    else:
        a = round(x / 5, 0) * 5
        b = round(x / 10, 0) * 10
        if np.abs(a - x) < np.abs(b - x):
            return a
        else:
            return b


def f_pvalueReport(tepred):
    x, y = tepred.P_value * 100, tepred.iloc[:, 1]
    rawbins = f_rawbins(x, y)  # 原始的分箱
    print('===原始分箱=====', rawbins)
    mhbins = [f_zcPvalue(i) for i in rawbins]
    mhbins = pd.Series(mhbins).unique().tolist()  # 1130发现 出现0的情况 解决方法
    mhbins[0] = 0
    mhbins[-1] = f_zcPvalue(max(rawbins) + 3.5)  # 加3.5的原因为 让71这种能corver到做大值
    bestbins = pd.Series(mhbins).unique().tolist()
    print('===调整后的分箱=====', bestbins)
    # 1201 开发一个方法 加上百分号,1204发现 加上百分号之后 顺序会发生改变，尝试另外的方法
    # 2018-01-08 发现 当数据量少时 分箱会出错，加入这样的逻辑 如果箱中只有一个元素 则等频分箱
    if len(bestbins) > 1:
        tepred['scoreBins'] = pd.cut(x, bestbins, right=False)
    elif len(bestbins) == 1 and x.min() == x.max():
        tepred['scoreBins'] = pd.cut(x, [x.min(), x.max() + 1], right=False)
    elif len(bestbins) == 1 and x.min() != x.max():
        try:
            tepred['scoreBins'] = pd.qcut(x, 5, duplicates='drop')
        except IndexError:  # 分箱不了
            tepred['scoreBins'] = pd.cut(x, [x.min(), x.max() + 1], right=False)

    iv = IV(tepred['scoreBins'], y)
    iv['bins'] = iv['bins'].apply(f_addbfh)
    model_pvalue = tepred.groupby('scoreBins')['P_value'].mean().tolist() + [tepred['P_value'].mean()]
    # model_pvalue.append(np.nan)
    iv['model_pvalue'] = model_pvalue
    # print(tepred.head(100))
    return iv


def f_addbfh(x):
    '''
    将区间里的概率值加上 % 号
    :param x: [0,5)
    :return: [0%,5%)
    '''
    # ls = []
    # for idx in range(len(x) - 1):
    #     _, __ = str(x[idx]), str(x[idx + 1])
    #     ls.append('[' + _ + '% , ' + __ + '%)')
    x = str(x)
    try:
        a, b = x.split(',')
        a = a + '%'
        b1, b2 = b.split(')')
        b1 = b1 + '%'
        ls = a + ' ,' + b1 + ')'
        return ls
    except ValueError:
        ls = x
        return ls


# 测试一下
# xx = pd.read_csv(
#     r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\ccxboost\model20171201204215\modeldata\d_2017-12-01_train_predict.csv')
# f_pvalueReport(xx)


#
# 测试一下这个函数
# dd = pd.read_csv(
#     r'C:/Users/liyin/Desktop/CcxMLOGE/TestUnit/ccxboost/model20171130203843/modeldata/d_2017-11-30_train_predict.csv')
# bins = [0.0, 10.0, 15.0, 20.0, 25.0, 35.0, 70.0]
# dd['scoreBins']=pd.cut(dd.P_value * 100, bins)
# dd.groupby('scoreBins')['P_value'].mean().tolist()


def f_modeldataInfo(repathlist, train, test, target_name):
    '''
    对切分出来的训练集和测试集 进行描述性的分析
    :param repathlist:
    :param trian:
    :param test:
    :return:
    {'入模维度': 61.0,
     '正负样本比': 5.9500000000000002,
     '总维度': 3145.0,
     '样本量': 9661.0,
     '正样本比例': 0.14380000000000001,
     '重要变量': 58.0}
    '''
    # 1.加载最终模型 计算出入模所需的变量数
    modelen = f_getmodelen(repathlist[1])
    # 2.总维度
    trrow, trcol = train.shape
    terow, tecol = test.shape
    # 3.重要变量个数
    implen = f_readdata(repathlist[2]).shape[0]
    # 4.正负样本的比例
    x = train[target_name].value_counts().values.tolist()
    y = train[target_name].value_counts().values.tolist()
    trpostivePct = x[1] / sum(x)
    tepostivePct = y[1] / sum(x)

    trnegdivpos = x[0] / x[1]
    tenegdivpos = y[0] / y[1]

    re = {'入模维度': [modelen, modelen],
          '正负样本比': [trnegdivpos, tenegdivpos],
          '总维度': [trcol, tecol],
          '样本量': [trrow, terow],
          '正样本比例': [trpostivePct, tepostivePct],
          '重要变量': [implen, implen]}
    return pd.DataFrame(re).to_dict(orient='records')


def f_getmodelen(model_path):
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
        try:
            # 随机森林的获取方法
            modellen = x[0].n_features_
        except:
            try:

                # gbm 获取入模变量的方法
                modellen = len(x.feature_name())
            except:
                # 评分卡 获取入模变量的方法
                modellen = len(x.params) - 1


    return modellen


def f_part5Output(repathlist, userPath, desres, modelres, impres, modelPath):
    '''
    第五部分的输出
    :param repathlist: 算法返回的列表
    :param userPath: 用户路径
    :param desres: 描述性分析的结果
    :param modelres: 模型输出结果
    :param impres: 模型输出的重要变量 part3计算出来
    :return:
    '''
    creattime = datetime.today().strftime('%Y-%m-%d_%H%M%S')
    filename = 'analysisReport' + '_' + creattime + '_Ccx' + '.xlsx'
    respath = f_mkdir(userPath, 'modelres')
    path = os.path.join(respath, filename)
    f_analysisReportMain(path, desres, modelres, impres)  # 保存结果至Excel
    part5 = {
        "predictResPath": [repathlist[3], repathlist[4]],  # 这个地方改为list
        "modelPath": modelPath,  # 1212 修改的 将模型保存下来
        "analysisReport": path
    }
    return part5


@ABS_log('MLogEDebug')
def f_type2Output(reqId, datasetInfo, descout, path, repathlist, rawdatacol, train, test, target_name, userPath,
                  desres, modelPath):
    part2 = dict({"datasetInfo": datasetInfo}, **descout)
    part3_2 = f_part3Output(repathlist[2], path, rawdatacol)  # bug 不能为train 因为train已经one-hot过了
    reqTime = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    part_4 = f_part4Output(repathlist, train, test, target_name, userPath)
    part_5 = f_part5Output(repathlist, userPath, desres, part_4, part3_2, modelPath)
    dict2 = {"reqId": reqId, "reqTime": reqTime, "type": 2,
             "dataDescription": part2,
             "variableAnalysis": part3_2,
             "modelOutput": part_4,
             "otherOutput": part_5,
             "optimizationOutput":repathlist[0]
             }

    return simplejson.dumps(dict2, ensure_ascii=False, ignore_nan=True, cls=MyEncoder)


def f_modelOutputWriter(writer, res, res_part3):
    '''
    将模型计算出的结果写出到Excel中
    :param write: 要写出的Excel writer 对象 接着变量描述的写入
    :param res: f_part4Output 函数计算出来的字典
    {"modeldataInfo": modeldataInfo, "modelreport": modelreport,
            "aucksPlot": aucksPlot, "pvalueReport": pvalueReport
            }
    :param res_part3: f_part3Output 函数计算出的结果
    {"impVar": impVar.to_dict(orient='records'), "topNpath": topNpath}
    :return: 保存成Excel
    '''
    modeldataInfo = pd.DataFrame(res['modeldataInfo'], index=['训练集', '测试集'])
    modelreport = pd.DataFrame(res['modelreport'], index=['训练集', '测试集'])
    pvalueReport = pd.DataFrame(res['pvalueReport'])

    PlottrainKS = pd.read_csv(res['aucksPlot']['trainKSpath'])
    PlottrainAUC = pd.read_csv(res['aucksPlot']['trainAUCpath'])
    PlottestKS = pd.read_csv(res['aucksPlot']['testKSpath'])
    PlottestAUC = pd.read_csv(res['aucksPlot']['testAUCpath'])

    # writer = pd.ExcelWriter(path)
    modeldataInfo.to_excel(writer, 'modeldataInfo')
    # 重要变量
    modelreport.to_excel(writer, 'modelreport')
    pvalueReport.to_excel(writer, 'pvalueReport', index=False)
    pd.DataFrame(res_part3['impVar']).to_excel(writer, 'ImpVar')
    f_readdata(res_part3['topNpath']).to_excel(writer, 'ImpVarIVdetail', index=False)

    PlottrainKS.to_excel(writer, 'PlottrainKS', index=False)
    PlottrainAUC.to_excel(writer, 'PlottrainAUC', index=False)
    PlottestKS.to_excel(writer, 'PlottestKS', index=False)
    PlottestAUC.to_excel(writer, 'PlottestAUC', index=False)

    writer.save()
    # writer.colse()
    print('模型计算结果保存完成')


# 测试一下
# res = {
#     "modeldataInfo": [
#         {
#             "入模维度": 87.0,
#             "总维度": 90.0,
#             "样本量": 17132.0,
#             "正样本比例": 0.18118141489610087,
#             "正负样本比": 4.519329896907217,
#             "重要变量": 52.0
#         },
#         {
#             "入模维度": 87.0,
#             "总维度": 90.0,
#             "样本量": 7343.0,
#             "正样本比例": 0.18118141489610087,
#             "正负样本比": 4.519329896907217,
#             "重要变量": 52.0
#         }
#     ],
#     "modelreport": [
#         {
#             "AUC": 0.7703199776808679,
#             "KS": 0.38654493296176423,
#             "f1-score": 0.17039343334276819,
#             "gini": 0.5406399553617358,
#             "precision": 0.7016317016317016,
#             "recall": 0.09697164948453608,
#             "support": 17132.0
#         },
#         {
#             "AUC": 0.7340033311284198,
#             "KS": 0.33224417271350665,
#             "f1-score": 0.17039343334276819,
#             "gini": 0.4680066622568395,
#             "precision": 0.7016317016317016,
#             "recall": 0.09697164948453608,
#             "support": 7343.0
#         }
#     ],
#     "aucksPlot": {
#         "trainKSpath": "C:/Users/liyin/Desktop/CcxMLOGE/TestUnit/trainKSpath.csv",
#         "trainAUCpath": "C:/Users/liyin/Desktop/CcxMLOGE/TestUnit/trainAUCpath.csv",
#         "testKSpath": "C:/Users/liyin/Desktop/CcxMLOGE/TestUnit/testKSpath.csv",
#         "testAUCpath": "C:/Users/liyin/Desktop/CcxMLOGE/TestUnit/testAUCpath.csv"
#     },
#     "pvalueReport": [
#         {
#             "bins": "[4.4, 9.956)",
#             "good": 1573,
#             "bad": 56,
#             "total": 1629,
#             "factor_per": 0.2218439329974125,
#             "bad_per": 0.03437691835481891,
#             "p": 0.042105263157894736,
#             "q": 0.261599866954931,
#             "woe": -1.8266433624993594,
#             "IV": 0.7635011593202796,
#             "varname": "P_value"
#         },
#         {
#             "bins": "[9.956, 16.0)",
#             "good": 1321,
#             "bad": 169,
#             "total": 1490,
#             "factor_per": 0.20291434018793408,
#             "bad_per": 0.11342281879194631,
#             "p": 0.12706766917293233,
#             "q": 0.21969067021453517,
#             "woe": -0.5475007397754827,
#             "IV": 0.7635011593202796,
#             "varname": "P_value"
#         },
#         {
#             "bins": "[16.0, 20.0)",
#             "good": 954,
#             "bad": 190,
#             "total": 1144,
#             "factor_per": 0.15579463434563529,
#             "bad_per": 0.1660839160839161,
#             "p": 0.14285714285714285,
#             "q": 0.158656244802927,
#             "woe": -0.1048947494640313,
#             "IV": 0.7635011593202796,
#             "varname": "P_value"
#         },
#         {
#             "bins": "[20.0, 29.0)",
#             "good": 1392,
#             "bad": 393,
#             "total": 1785,
#             "factor_per": 0.2430886558627264,
#             "bad_per": 0.22016806722689075,
#             "p": 0.2954887218045113,
#             "q": 0.23149842008980542,
#             "woe": 0.24405762079866558,
#             "IV": 0.7635011593202796,
#             "varname": "P_value"
#         },
#         {
#             "bins": "[29.0, 76.0)",
#             "good": 773,
#             "bad": 522,
#             "total": 1295,
#             "factor_per": 0.17635843660629172,
#             "bad_per": 0.40308880308880307,
#             "p": 0.3924812030075188,
#             "q": 0.12855479793780142,
#             "woe": 1.1161333891189862,
#             "IV": 0.7635011593202796,
#             "varname": "P_value"
#         },
#         {
#             "bins": "All",
#             "good": 6013,
#             "bad": 1330,
#             "total": 7343,
#             "factor_per": 1.0,
#             "bad_per": 0.1811248808388942,
#             "p": 1.0,
#             "q": 1.0,
#             "woe": 0.0,
#             "IV": 0.7635011593202796,
#             "varname": "P_value"
#         }
#     ]
# }
#
# f_modelOutputWriter(r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\modelOutput.xlsx', res)


def f_analysisReportMain(path, desres, modelres, impres):
    '''
    存储计算结果的主函数
    :param path: 写入的Excel的路径 xxx.xlsx
    :param desres: 描述性分析的结果 f_mainDesc 计算出来
    :param modelres: 模型计算出的结果 f_part4Output 函数计算
    :return: 写入指定路径下的Excel文件中
    '''

    writer = f_VardescWriter(path, desres)
    f_modelOutputWriter(writer, modelres, impres)
    print('项目计算结果已保存至：', path)


# 测试一下 Excel 会被覆盖的问题
# r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\analysisReport_2017-11-30_175927_Ccx.xlsx'


def f_mkdir(userPath, dir):
    '''
    一个函数 在用户目录下 创建文件夹
    ccxboost,ccxrf,ccxgbm,conf,datafile, vardesc,modelres

    :param userPath:
    :return: 新建文件夹的绝对路径
    '''
    path = os.path.join(userPath, dir)
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    return path


# 模型预测结果返回--无监督返回
# 请求格式
'''
{'reqId': '请求id', 'modelreqId': '模型训练时的请求id', 'modelPath': '模型保存的路径',
 'base': '待预测的数据集位置'
 }
'''

'''
返回格式
正常：
{'code':'','reqId':'','predictResPath':'预测结果的文件路径','reqTime':'请求时间戳'}
异常：
{'code':'503','Msg':'异常信息'}
'''


def f_modelPredictOutputType0(reqId, predictResPath):
    reqTime = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    res = {'code': 200, 'reqId': reqId, 'predictResPath': predictResPath, 'reqTime': reqTime}
    return simplejson.dumps(res, ensure_ascii=False)


# 模型预测结果返回--有监督返回
# 请求格式 和无监督保持一致
'''
{'reqId': '请求id', 'modelreqId': '模型训练时的请求id', 'modelPath': '模型保存的路径',
 'base': '待预测的数据集位置'
 }
'''

'''
返回格式 --内容比无监督多很多
正常：
{'code':'状态码','reqId':'请求id','predictResPath':'预测结果的文件路径','reqTime':'请求时间戳',
'modeldataInfo':'预测集数据预览','modelreport':'预测集模型指标报告','pvalueReport':'预测集概率分箱结果',
}

"modeldataInfo": [
      {
        "入模维度": 87.0,
        "总维度": 90.0,
        "样本量": 17103.0,
        "正样本比例": 0.18096240425656318,
        "正负样本比": 4.526009693053312,
        "重要变量": 49.0
      }]
      
"modelreport": [
      {
        "AUC": 0.7659502555198091,
        "KS": 0.38199279617739784,
        "f1-score": 0.18100056211354693,
        "gini": 0.5319005110396182,
        "precision": 0.6954643628509719,
        "recall": 0.10403877221324717,
        "support": 17103.0
      }]

"pvalueReport": [
      {
        "bins": "[0.0% , 10.0%)",
        "good": 3479,
        "bad": 94,
        "total": 3573,
        "factor_per": 0.20891071741799686,
        "bad_per": 0.026308424293310942,
        "p": 0.03037156704361874,
        "q": 0.24835808109651628,
        "woe": -2.101364704034947,
        "IV": 0.9909,
        "varname": "scoreBins",
        "model_pvalue": 0.05899743113970176,
        "modelDataName": "train"
      }]
      
异常：
{'code':'503','Msg':'异常信息'}
'''


def f_modelPredictOutputType1(reqId, predictResPath, modelPath, predict, base):
    '''
    有监督预测的返回结果
    :param reqId:
    :param predictResPath:预测结果的存储路径
    :param modelPath:模型路径
    :param predict:预测集
    :param base:传入的base
    :return:
    '''
    reqTime = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    # modeldataInfo
    # 1.加载最终模型 计算出入模所需的变量数
    modelen = f_getmodelen(modelPath)
    # 2.总维度
    trrow, trcol = predict.shape
    # 3.重要变量个数
    implen = None  # 前台补充一下吧
    # 4.正负样本的比例
    target_name = base['targetName']
    x = predict[target_name].value_counts().values.tolist()
    trpostivePct = x[1] / sum(x)
    trnegdivpos = x[0] / x[1]

    re = {'入模维度': modelen,
          '正负样本比': trnegdivpos,
          '总维度': trcol,
          '样本量': trrow,
          '正样本比例': trpostivePct,
          '重要变量': implen,
          '算法': base['arithmetic'],
          '模型配置': base['modelConf']}
    modeldataInfo = re

    # modelreport
    trpredpath = predictResPath
    trpred = pd.read_csv(trpredpath)
    trpred.rename(columns={'predictProb': 'P_value'}, inplace=True)

    trp = precision_score(trpred[target_name], trpred.P_value > 0.5, average='binary')
    trr = recall_score(trpred[target_name], trpred.P_value > 0.5, average='binary')
    trf1score = f1_score(trpred[target_name], trpred.P_value > 0.5, average='binary')
    trauc, trks = f_getAucKs(trpredpath)
    trgini = trauc * 2 - 1
    trsupport = trpred.shape[0]

    re = {'AUC': trauc,
          'KS': trks,
          'f1-score': trf1score,
          'gini': trgini,
          'precision': trp,
          'recall': trr,
          'support': trsupport}

    modelreport = re

    # pvalueReport
    prepvalueReport = f_pvalueReport(trpred)
    prepvalueReport['modelDataName'] = 'predict'
    prepvalueReport = prepvalueReport.to_dict(orient='records')

    res = {'code': 200, 'reqId': reqId, 'predictResPath': predictResPath, 'reqTime': reqTime,
           'modeldataInfo': modeldataInfo, 'modelreport': modelreport, 'pvalueReport': prepvalueReport}
    return simplejson.dumps(res, ensure_ascii=False)
