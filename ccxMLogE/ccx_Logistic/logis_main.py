import pickle,os
from flask import jsonify
import time

from ccxMLogE.ccx_Logistic.ModelTrain import *
from ccxMLogE.ccx_Logistic.fill_na import FillNan
from ccxMLogE.ccx_Logistic.testData_predict import clean_testData, testData_woe, predict
from ccxMLogE.ccx_Logistic.var_filter import data_split, woeTransData, Varfilter, filterMultiCol, filterStep
from ccxMLogE.ccx_Logistic.dataTrans import datatrans, saveTransData

def transData(varData,targetData,path,null_up, null_down, var_manu,fillNumType, fillClassType,BinNum,BinClass,mllog):
    try:
        mllog.info('特征清洗开始')
        t0 = time.time()
        train_data, train_target, test_data, test_target = data_split(varData, targetData, split_ratio=0.3)
        with open(os.path.join(path, 'pkl/target.pkl'), 'wb') as f:
            pickle.dump(train_target, f)
        with open(os.path.join(path, 'pkl/test_target.pkl'), 'wb') as f:
            pickle.dump(test_target, f)
        with open(os.path.join(path, 'pkl/test_data.pkl'), 'wb') as f:
            pickle.dump(test_data, f)
        fillData = FillNan(path)
        res = fillData.mergeData(train_data, null_up, null_down, var_manu, 'missing', fillNumType, fillClassType)
        fillData.saveData()
        mllog.info('变量缺失值处理完毕')
        t1 = time.time()
        mllog.info('变量缺失值处理耗时：%s 秒'%(t1-t0))
        rawTrainData, varType = res[0], res[1]  # 清洗好以后的数据,变量类型
        _ = woeTransData(path, rawTrainData, train_target, varType, BinNum, True, BinClass)  # 这一步慢
        mllog.info('woe初步转换完毕')
        mllog.info('woe初步转换耗时：%s 秒' % (time.time() - t1))
        json_str = {'varType':varType,'iv_path':os.path.join(path,'ivRes')}
        return json_str
    except Exception as e:
        return jsonify({'code':500,'error message':str(e)})


def logisModelTrain(thres_iv,modelpath,thres_vif,steptype,varType,mllog):
    try:
        t0 = time.time()
        varf = Varfilter(threshold=thres_iv)
        var2woe, iv, train_target, test_data, test_target = read_savedata(modelpath)
        var2woe2, dict_filt = varf.filterIV(iv, var2woe)  # iv值筛选后，并转成了woe的变量
        var_vifilter = filterMultiCol(var2woe2, thres=thres_vif)  # vif 筛选变量
        var_vifilter = var_vifilter.reindex(train_target.index)
        step_res, step_var = filterStep(var_vifilter, train_target, steptype)  # 逐步回归筛选变量
        clf, var_sel, df_res = lr_fit(modelpath, step_var, train_target)  # 剔除掉系数为负的变量

        cleanTest = clean_testData(modelpath, test_data)  # 对测试数据进行清洗
        cols_class = list(set(varType[0]).intersection(cleanTest[0].columns))
        cols_num = list(set(varType[1]).intersection(cleanTest[0].columns))
        transData = testData_woe(modelpath, cleanTest[0], cols_num, cols_class)  # 测试集数据转换
        predict_res = predict(modelpath, clf, df_res, var_sel, transData, train_target, test_target)  # 测试集的概率转换
        mllog.info('模型训练耗时： %s 秒'%(time.time()-t0))

        psd = datatrans(modelpath, varType)  # 数据转换类
        saveMpath = saveTransData(modelpath, psd)
        predict_res['otherOutput']['modelPath'] = saveMpath  # 更换modelpath值



        return predict_res
    except Exception as e:
        return jsonify({'code':500,'error message': str(e)})