import flask
from flask import request
from flask import jsonify
from sklearn.externals import joblib
import json
from ccxMLogE.ccx_Logistic.var_filter import *
from ccxMLogE.ccx_Logistic.fill_na import FillNan
from ccxMLogE.ccx_Logistic.cut_bins import *
from ccxMLogE.ccx_Logistic.ModelTrain import *
from ccxMLogE.ccx_Logistic.testData_predict import clean_testData,testData_woe,predict,ModelAuc
from ccxMLogE.ccx_Logistic.Logis_mkdir import Logis_mkdir
global res,train_target,var2woe,varType,iv,test_data,test_target
global base,rawdata,cateList,userPath
iv={}
server = flask.Flask(__name__)
@server.route('/ccxModelApi/ccxlogistic', methods=['post'])
def preprocess():
    try:
        path = Logis_mkdir(userPath,base) # 第一次跑逻辑回归项目时生成子文件夹
        json_data = json.loads(request.data.decode())
        id,target,var_manu = base['index'],base['targetName'],cateList # id，目标变量，人为指定分类变量
        null_up,null_down = json_data['NullUp'],json_data['NullDown'] # 缺失值比例上下限
        fillNumType, fillClassType = json_data['FillNumType'],json_data['FillClassType'] # 连续变量填充类型，离散变量填充类型
        BinNum,BinClass = json_data['BinNum'],json_data['BinClass'] # 变量分箱方法
        # rawData
        varData,targetData = rawdata.drop(target,axis=1),rawdata[target]

        train_data,train_target,test_data,test_target = data_split(varData, targetData, split_ratio=0.3)

        fillData = FillNan(path)
        res = fillData.mergeData(train_data,null_up,null_down,var_manu,'missing',fillNumType,fillClassType)
        rawTrainData,varType = res[0],res[1] # 清洗好以后的数据,变量类型
        cur_path,_ = os.getcwd(),os.chdir(path)
        var2woe, iv = woeTransData(rawTrainData,train_target,varType,BinNum,True,BinClass) # 这一步慢
        os.chdir(cur_path)
    except Exception as e:
        return jsonify({"code": 500, "msg": "计算失败", "error_msg": str(e)})


@server.route('/ccxModelApi/rebin', methods=['post'])
def rebin():
    path = Logis_mkdir(userPath)
    cur_path, _ = os.getcwd(), os.chdir(path)
    json_data = json.loads(request.data.decode())
    varName,bins_num,bins_class = json_data['VarName'],json_data['CustBinNum'],json_data['CustBinClass'] # 变量名,分割点
    var = res[0][varName] # 变量所在列
    cur_path, _ = os.getcwd(), os.chdir(path)
    if varName in varType[1]: # 连续变量
        try:
            var2 = var[var!='missing'] # 剥离出变量值不为missing的样本
            missValue = var[var=='missing']
        except Exception as e:
            var_str = var.astype(str)
            var2 = var[var_str!='missing']
            missValue = var[var_str == 'missing']
        iv,woe_col = IV_numeric(var2,train_target,missValue,bins_num,isdump=True)

    elif varName in varType[0]: # 离散变量
        var2 = var.apply(lambda i: bins_class[i])
        iv, woe_col = IV_class(var2, train_target, isdump=True)
    var2woe[var] = woe_col
    os.chdir(cur_path)
    return '自定义分箱已经完成'

@server.route('/modeltrain', methods=['post'])
def LRModel():

    try:
        json_data = json.loads(request.data.decode())
        thres_iv, thres_vif = json_data['ThresIv'], json_data['ThresVif']  # vif 阈值
        steptype = json_data['StepType']  # 逐步回归类型
        varf = Varfilter(threshold=thres_iv)
        var2woe2, dict_filt = varf.filterIV(iv, var2woe)  # iv值筛选后，并转成了woe的变量
        var_vifilter = filterMultiCol(var2woe2, thres=thres_vif)  # vif 筛选变量
        var_vifilter = var_vifilter.reindex(train_target.index)
        step_res, step_var = filterStep(var_vifilter, train_target, steptype)  # 逐步回归筛选变量
        clf, var_sel = lr_fit(step_var, train_target)  # 剔除掉系数为负的变量

        cleanTest = clean_testData(test_data)  # 对测试数据进行清洗
        cols_class = list(set(varType[0]).intersection(cleanTest[0].columns))
        cols_num = list(set(varType[1]).intersection(cleanTest[0].columns))
        transData = testData_woe(cleanTest[0], cols_num, cols_class)  # 测试集数据转换
        train_auc,test_auc = predict(clf,var_sel,transData,train_target,test_target)  # 测试集的概率转换
    except Exception as e:
        return jsonify({'code':500,'error_message': str(e)})

@server.route('/scorecard', methods=['post'])
def scorecard():
    json_data = json.loads(request.data.decode())
    baseScore, pdo = json_data['BaseScore'], json_data['PDO']  # 评分卡的参数
    A, B = get_AB(baseScore, pdo)  # 得到A，B系数
    with open('model/LR.model','rb') as f: clf = joblib.load(f)
    with open('model/var_model.pkl','rb') as f: var = joblib.load(f)
    df = scoreCardOut(clf, var, A, B)  # 得到评分卡
    df.to_csv('model/scorecard.csv')












