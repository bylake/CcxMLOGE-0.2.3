
from ccxMLogE.ccx_Logistic.cut_bins import *
from ccxMLogE import outputTransform
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_score,recall_score,f1_score
import simplejson

def path_merge(path, fname):
    fpath = os.path.join(path, fname)
    return fpath


def clean_testData(path, test_data):
    '''
    对测试集数据进行清洗
    :param test_data:  测试集
    :return: 清洗好以后的数据
    '''
    with open(path_merge(path, 'model/LR_model.pkl'), 'rb') as f:
        var_sel = pickle.load(f).params.index.tolist()  # read model var
        var_sel.remove('intercept')
    with open(path_merge(path, 'pkl/var_missing.pkl'), 'rb') as f:
        miss_var = pickle.load(f)  # read missing var
    with open(path_merge(path, 'pkl/number_na.pkl'), 'rb') as f:
        num_var = pickle.load(f)  # read num var
    with open(path_merge(path, 'pkl/str_na.pkl'), 'rb') as f:
        class_var = pickle.load(f)  # read class var
    test_data = test_data[var_sel]

    com_var_miss = set(test_data.columns).intersection(set(miss_var))  # missing var fill
    if len(com_var_miss) > 0:
        test_data[list(com_var_miss)] = test_data[list(com_var_miss)].fillna('missing')

    com_var_num = set(test_data.columns).intersection(num_var.index)  # num var fill
    if len(com_var_num) > 0:
        fillValue = num_var.loc[list(com_var_num)]
        test_data[list(com_var_num)] = test_data[list(com_var_num)].fillna(fillValue)

    com_var_class = set(test_data.columns).intersection(class_var.index)  # class var fill
    if len(com_var_class) > 0:
        fillValue = class_var.loc[list(com_var_class)]
        test_data[list(com_var_class)] = test_data[list(com_var_class)].fillna(fillValue)

    return test_data, com_var_num, com_var_class, com_var_miss


def testData_woe(path, test_data, var_num, var_class):
    '''
    完成对测试集的woe转换
    :param test_data:
    :param var_num:
    :param var_class:
    :return: woeData
    '''
    num_woe = pd.concat(map(lambda i: get_woe_num(path, i, test_data), var_num), axis=1)
    class_woe = pd.concat(map(lambda i: get_woe_class(path, i, test_data), var_class), axis=1)
    woeData = pd.concat([num_woe, class_woe], axis=1)
    woeData2 = woeData.dropna(how='any')
    return woeData2

@ABS_log('predict')
def predict(modelpath, clf, df_res, train_woeData, test_woeData, train_target, test_target):
    '''
    :param woeData:
    :return: 概率
    '''
    # with open('model/LR_model.pkl', 'rb') as f: clf = pickle.load(f)  # 保存模型
    train_woeData['intercept'] = 1
    test_woeData['intercept'] = 1
    prob_bad_train = pd.Series(clf.predict(train_woeData), index=train_woeData.index)
    prob_bad_test = pd.Series(clf.predict(test_woeData), index=test_woeData.index)
    test_target2 = test_target.reindex(prob_bad_test.index)  # 测试集标签索引调整

    train_res = pd.concat([prob_bad_train,train_target],axis=1)
    test_res = pd.concat([prob_bad_test,test_target],axis=1)
    path_train = os.path.join(modelpath,'model/train_res.csv')
    path_test = os.path.join(modelpath,'model/test_res.csv')
    train_res.to_csv(path_train),test_res.to_csv(path_test) # 训练集和测试集预测概率和标签
    model_train_res,df_train_ks,train_iv = modelResult(prob_bad_train,train_target) # 训练集的模型结果
    model_test_res,df_test_ks,test_iv = modelResult(prob_bad_test,test_target2) # 测试集的模型结果

    des_model = pd.concat([model_train_res.rename('train'),model_test_res.rename('test')],axis=1).T # model report
    model_iv = pd.concat([train_iv,test_iv])
    # model_coef = pd.Series(clf.coef_[0], index=train_woeData.columns)
    # model_intercept = pd.Series([clf.intercept_[0]], index=['intercept'])
    # model_A_coef = pd.concat([model_intercept, model_coef])
    writer = pd.ExcelWriter(os.path.join(modelpath, 'model/anaylysisRep.xlsx'))
    des_model.to_excel(writer,'modelreport')
    model_iv.to_excel(writer,'pvalueReport')
    df_train_ks.to_excel(writer, 'PlottrainKs')
    df_train_ks.loc[:, :2].to_excel(writer, 'PlottrainAUC', index=False)
    df_test_ks.to_excel(writer, 'PlottestKs')
    df_test_ks.loc[:, :2].to_excel(writer, 'PlottestAUC', index=False)

    # res = model_A_coef.reset_index()
    # res.columns=['varName','coef']
    df_res.to_excel(writer, 'ModelCoef', index=False)

    path_dict = path_ks(modelpath,df_train_ks,df_test_ks) # train,test auc path
    pvalueReport = model_iv.to_dict(orient='records') # pvalue report
    modelinfo_dict = f_logis_modeldataInfo(train_woeData,test_woeData,train_target,test_target)

    df_modelinfo = pd.DataFrame(modelinfo_dict,index=['训练集','测试集'])

    df_modelinfo.to_excel(writer,'modeldatainfo')
    modelreport = des_model.to_dict(orient='records')

    writer.save()
    # return simplejson.dumps({"modeldataInfo": modelinfo_dict, "modelreport": modelreport,
    #         "aucksPlot": path_dict, "pvalueReport": pvalueReport
    #         },ensure_ascii=False, ignore_nan=True, cls=MyEncoder)
    json1 = {"modeldataInfo": modelinfo_dict, "modelreport": modelreport,
            "aucksPlot": path_dict, "pvalueReport": pvalueReport
            }
    ReportPath = os.path.join(modelpath,'model/anaylysisRep.xlsx')
    ModelPath = os.path.join(modelpath,'model/LR_model.pkl')
    json2 = {'predictResPath':[path_train,path_test],'modelPath':ModelPath,'analysisReport':ReportPath}
    # json3 = res.to_dict(orient='records')
    path_iv_imp = []
    for i in test_woeData.columns:
        path_iv_imp.append(os.path.join(modelpath,'ivRes_%s.csv'%i))

    return {'modeloutput': json1, 'otherOutput': json2, 'modelCoef': os.path.join(modelpath, 'model/var_summary.csv'),
            'modelVariable': path_iv_imp}



def path_ks(modelpath,df_train_ks,df_test_ks):
    path_train_1 = os.path.join(modelpath,'model/PlottrainKs.csv')
    path_train_2 = os.path.join(modelpath,'model/PlottrainAUC.csv')
    path_test_1 = os.path.join(modelpath,'model/PlottestKs.csv')
    path_test_2= os.path.join(modelpath,'model/PlottestAUC.csv')
    df_train_ks.to_csv(path_train_1,index=False)
    df_train_ks.loc[:, :2].to_csv(path_train_2,index=False)
    df_test_ks.to_csv(path_test_1,index=False)
    df_test_ks.loc[:, :2].to_csv(path_test_2,index=False)
    aucksPlot = {'trainKSpath': path_train_1,
                 'trainAUCpath': path_train_2,
                 'testKSpath': path_test_1,
                 'testAUCpath': path_test_2,
                 }

    return aucksPlot

def f_logis_modeldataInfo(train, test, train_target,test_target):
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
    # 2.总维度
    trrow, trcol = train.shape
    terow, tecol = test.shape
    # 3.重要变量个数
    # 4.正负样本的比例
    x = train_target.value_counts().values.tolist()
    y = test_target.value_counts().values.tolist()
    trpostivePct = x[1] / sum(x)
    tepostivePct = y[1] / sum(y)

    trnegdivpos = x[0] / x[1]
    tenegdivpos = y[0] / y[1]

    res = {
          '正负样本比': [trnegdivpos, tenegdivpos],
          '入模维度': [trcol, tecol],
          '样本量': [trrow, terow],
          '正样本比例': [trpostivePct, tepostivePct],
          }
    return res

def modelResult(prob_data,target):
    '''

    :param prob_data:
    :param target:
    :return: ks,auc,gini,f1score,precision,recall_rate,support
    '''
    fpr_train, tpr_train, ks_dis_train, ks_train, auc_train = ModelAuc(prob_data, target)
    df_train_ks = pd.DataFrame(
        np.hstack([fpr_train[:, np.newaxis], tpr_train[:, np.newaxis], ks_dis_train[:, np.newaxis]]),
        index=np.arange(len(fpr_train)))

    trp = precision_score(target, prob_data > 0.5, average='binary')
    trr = recall_score(target, prob_data > 0.5, average='binary')
    trf1score = f1_score(target, prob_data > 0.5, average='binary')
    trgini = auc_train * 2 - 1
    trsupport = target.shape[0]

    res = pd.Series({'AUC': auc_train,
          'KS': ks_train,
          'f1-score': trf1score,
          'gini': trgini,
          'precision': trp,
          'recall': trr,
          'support': trsupport})
    pData = pd.concat([prob_data,target],axis=1)
    pData.columns=['P_value','target']
    model_iv = outputTransform.f_pvalueReport(pData)

    return res,df_train_ks,model_iv





def trans_score(woeData, A, B):
    '''

    :param woeData:
    :param A:
    :param B:
    :return: 分数
    '''
    with open('model/LR_model.pkl', 'rb') as f: clf = pickle.load(f)  # 保存模型
    proba_bad = clf.predict_proba(woeData)[:, 1]
    score = list(map(lambda i: A - B * np.log(i / (1 - i)), proba_bad))
    return score


def ModelAuc(proba_test, test_target):
    '''

    :param clf:
    :param test_data:
    :param test_target:
    :return: tpr,fpr
    '''
    test_target2 = test_target.reindex(proba_test.index)
    fpr, tpr, thresholds = roc_curve(test_target2, proba_test)
    auc = roc_auc_score(test_target2, proba_test)
    ks = tpr - fpr
    # df_ks =
    return fpr, tpr, ks, max(ks), auc
