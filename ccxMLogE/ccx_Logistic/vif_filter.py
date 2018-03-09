from statsmodels.stats.outliers_influence import variance_inflation_factor

class filter_vif():
    def __init__(self,thresh=10):
        self.thresh = thresh
    def calculate_vif(self,data):
        #data为数据集，pd.DataFrame
        varnames=[x for x in data.columns if x not in ['target','id']]
        X=data[varnames] #X为特征变量集合,pd.DataFrame
        dropped=True
        while dropped:
            variables = X.columns
            dropped=False
            vif = []
            for var in X.columns:
                new_vif = variance_inflation_factor(X[variables].values, X.columns.get_loc(var))
                #X.columns.get_loc(var)是指变量在第几列
                vif.append(new_vif)  #得出自变量VIF值的集合
                #if np.isinf(new_vif):如果有缺失值，则填充为0
                   # break
            max_vif = max(vif)
            if max_vif > self.thresh:  #如果共线性变量的VIF值大于指定的阈值
                maxloc = vif.index(max_vif)
                #print( X.columns[maxloc] )
                #print( max_vif )
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)  #去掉多重共线性的变量
                dropped=True
        return X

