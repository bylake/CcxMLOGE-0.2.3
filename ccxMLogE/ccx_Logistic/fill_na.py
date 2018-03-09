import pickle
import pandas as pd
from ccxMLogE.ccx_Logistic.VarType import DataType
import os


def get_mode(df):
    '''

    :param df:
    :return: 众数
    '''
    mode = df.value_counts().sort_values(ascending=False).index[0]
    return mode

def nullStat_get(data,UpperLmt,LowerLmt):
    '''
    缺失值统计
    :param data:
    :param UpperLmt:
    :param LowerLmt:
    :return:
    '''
    null_stat = data.isnull().sum()
    var1 = null_stat[null_stat/len(data)>UpperLmt] # 需要删除的变量
    var2 = null_stat[(null_stat/len(data)>=LowerLmt) & (null_stat/len(data)<=UpperLmt)] # 需要填充missing的变量，作为独立分箱
    var3 = null_stat[null_stat/len(data)<=LowerLmt] # 需要填充缺失值的变量
    return list(var1.index),list(var2.index),list(var3.index)

class FillNan():
    def __init__(self,path):
        self.number_na = 0
        self.str_na = 0
        self.missVar = 0
        self.path = path
    # 缺失值用missing填充
    def fill_missing(self,df_num,fillValue = 'missing'):
        df_num = df_num.fillna(fillValue)
        return df_num
    #连续变量，中位数填充
    def fill_median(self,df_num,type):
        if type==1: # 中位数填充
            self.number_na = df_num.median()
            df_num_fillna = df_num.fillna(self.number_na)

        elif type==2: # 均值填充
            self.number_na = df_num.mean()
            df_num_fillna = df_num.fillna(self.number_na)
        elif type == 3:  # 其他数
            self.number_na = df_num.apply(lambda i: get_mode(i), axis=0)
            df_num_fillna = df_num.fillna(self.number_na)
        return df_num_fillna

    #分类变量，众数填充
    def fill_mode(self,df_str,classType):
        #df: pd.Series
        value=df_str.apply(lambda i:get_mode(i),axis=0) #当变量的众数存在多个值时，默认选择以第一个值为众数值
        df_str=df_str.fillna(value)
        self.str_na=value
        return df_str
    def mergeData(self,data,UpperLmt,LowerLmt,var_manu,fillValue,numType,classType):
        '''

        :param data:
        :param UpperLmt:
        :param LowerLmt:
        :param var_manu:
        :param fillValue:
        :param numType: 数值变量填充类型
        :param classType: 离散变量填充类型

        :return: 缺失值填充以后的数据，以及变量类型
        '''
        v1,v2,v3 = nullStat_get(data,UpperLmt,LowerLmt)
        varData = data.drop(v1,axis=1) # 删除缺失值过高的变量
        vartype = DataType()
        vartype = vartype.f_VarTypeClassfiy(varData, var_manu)  # 获得变量类型
        varData_p1 = self.fill_missing(varData[v2],fillValue)
        var_class,var_num = set(vartype[0]).difference(set(v2)),set(vartype[1]).difference(set(v2))

        varData_class,varData_num = varData[list(var_class)],varData[list(var_num)]

        varData_num = self.fill_median(varData_num,numType)
        varData_class = self.fill_mode(varData_class,classType)
        self.dump_num() # 保存填充值为均值，众数类型的变量
        with open(os.path.join(self.path,'pkl/var_drop.pkl'),'wb') as f: pickle.dump(v1,f) # 保存扔掉的变量
        with open(os.path.join(self.path,'pkl/var_missing.pkl'),'wb') as f: pickle.dump(v2,f) # 保存填充为missing的变量
        self.merData = pd.concat([varData_class,varData_num],axis=1) # 离散型
        self.merData = pd.concat([self.merData,varData_p1],axis=1)
        return self.merData,vartype
    def saveData(self):
        # 保存清洗好的数据
        data_path = os.path.join(self.path, 'pkl/mergeData.pkl')
        with open(data_path,'wb') as f: pickle.dump(self.merData,f)





    def dump_num(self):
        '''
        # 保存缺失值填充信息
        :param number_na:
        :param str_na:
        :return:
        '''
        number_path = os.path.join(self.path,'pkl/number_na.pkl')
        str_path = os.path.join(self.path,'pkl/str_na.pkl')
        with open(number_path,'wb') as f:
            pickle.dump(self.number_na,f)
        with open(str_path,'wb') as f:
            pickle.dump(self.str_na,f)

    def load_num(self):
        # 导入缺失值填充信息
        number_path = os.path.join(self.path,'pkl/number_na.pkl')
        str_path = os.path.join(self.path,'pkl/str_na.pkl')
        miss_path = os.path.join(self.path,'pkl/var_missing.pkl')
        with open(number_path,'rb') as f:
            self.number_na = pickle.load(f)
        with open(str_path,'rb') as f:
            self.str_na = pickle.load(f)
        with open(miss_path, 'rb') as f:
            self.missVar = pickle.load(f)  # 保存填充为missing的变量
        return self.number_na,self.str_na,self.missVar

# data = pd.read_excel(r'E:\workproject\机器学习平台\rsc\sam_data_5w_1225_encode.xlsx',encoding='gbk').set_index('id')
# vartype=DataType_recog()
#
# vartype = vartype.f_VarTypeClassfiy(data.drop(['flag_person'],axis=1),[])
# fillData =  FillNan()
# Data1 = fillData.fill_median(data[vartype[1]])
# Data2 = fillData.fill_mode(data[vartype[0]])
# fillData.dump_num(fillData.number_na,fillData.str_na) # 保存缺失值填充数据
# d1,d2 = fillData.load_num()

# fill_data = pd.concat([Data1,Data2],axis=1)
# fill_data = pd.concat([fill_data,data.flag_person],axis=1)
