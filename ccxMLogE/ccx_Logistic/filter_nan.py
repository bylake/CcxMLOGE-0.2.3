def filter_nan(df,thres):
    '''
    删除掉缺失值比例大于阈值的数据和变量
    :param df:
    :return:
    '''
    null_row = df.isnull().sum(axis=1) # 按行统计缺失值
    null_col = df.isnull().sum(axis=0) # 按列统计缺失值
    df2 = df.loc[null_row<=df.shape[0]*thres,:]
    df3 = df2.loc[:,null_col<=df.shape[1]*thres]

    return df3