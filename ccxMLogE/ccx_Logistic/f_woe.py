"""
对多列特征变量进行woe转换,返回woe化的特征变量集合
data, pd.DataFrame类型
varnames,特征变量名称集合，list列表类型，
target，目标变量名称，字符类型
"""
def woe_trans(data,varnames,target):
    x_woe=[]
    for feature in varnames:
        x=data[feature]
        y=data[target]
        df=pd.crosstab(x,y,margins=True)
        df.columns=['good','bad','total']
        df['factor_per']=df['total']/len(y)
        df['bad_per']=df['bad']/df['total']
        df['p']=df['bad']/df.ix['All','bad']
        df['q']=df['good']/df.ix['All','good']
        df['woe']=np.log(df['p']/df['q'])
        df_new=df.reset_index()[[feature,'woe']]
        woe_dict={}
        for x in data.birthY.unique():
            woe_value=df_new.loc[df_new[feature]==x,'woe'].values
            if woe_value :
                woe_dict[x]=df_new.loc[df_new[feature]==x,'woe'].values[0]
            else:
                woe_dict[x]=np.nan
        data[feature+'_woe']=data[feature].map(woe_dict)
        x_woe.append(data[feature+'_woe'])
    data_woe=pd.concat(x_woe,axis=1)
    return data_woe