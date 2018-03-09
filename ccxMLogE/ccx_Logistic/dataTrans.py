from ccxMLogE.ccx_Logistic.testData_predict import clean_testData, testData_woe
import pickle, os


class datatrans():
    def __init__(self, modelpath, varType):
        self.modelpath = modelpath
        self.varType = varType
        self.bstmodelpath = os.path.join(modelpath, 'model/LR_model.pkl')
        with open(self.bstmodelpath, 'rb') as f: self.bstmodel = pickle.load(f)

        self.modelname = 'ccxscorecard'

    def getdata(self, test_data):
        cleanTest = clean_testData(self.modelpath, test_data)  # 对测试数据进行清洗
        cols_class = list(set(self.varType[0]).intersection(cleanTest[0].columns))
        cols_num = list(set(self.varType[1]).intersection(cleanTest[0].columns))
        transData = testData_woe(self.modelpath, cleanTest[0], cols_num, cols_class)  # 测试集数据转换
        return transData

    def getbstmodel(self):
        return self.bstmodel

    def getmodelname(self):
        return self.modelname


def saveTransData(modelpath, psd):
    '''
    保存数据清洗过程的对象
    :param modelpath:
    :param psd:
    :return:
    '''
    path = os.path.join(modelpath, 'model/transdata.pkl')
    if os.path.exists(path):
        pass
    else:
        with open(path, 'wb') as f:
            pickle.dump(psd, f)

    return path
