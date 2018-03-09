import pandas as pd
from sklearn.cluster import KMeans
def classMerge(data,y):

    '''
    kmeans merge variable
    :param data:
    :param i:
    :param y:
    :return: dict_class
    '''
    crtab = pd.crosstab(data, y, margins=True).drop('All')
    crtab.columns = ['good', 'bad', 'total']
    crtab['bad_ratio'] = crtab['bad']/crtab['total']
    kmeans = KMeans(n_clusters=5, random_state=0).fit(crtab[['good','bad','bad_ratio']])
    dict_class = dict(zip(crtab.index.tolist(),kmeans.labels_))
    return dict_class


def var_rank(data, feature, target, threshold=5):
    '''
    违约率分箱
    :param data: series
    :param feature:
    :param target:
    :param threshold:
    :return:
    '''

    data0 = data[[feature, target]].groupby(feature)[target].agg(['sum', 'count'])
    data0['default_bad'] = data0['sum'] / data0['count']
    data_sort = data0.sort_values(by='default_bad')
    var_list = pd.cut(data_sort['default_bad'], threshold).value_counts().sort_index(ascending=True).cumsum().tolist()
    var_list.insert(0, 0)
    var_index = data_sort.index.tolist()
    var_set = []
    for i in range(len(var_list) - 1):
        var_set.append(var_index[var_list[i]:var_list[i + 1]])
    dict_var = {}
    for i, k in enumerate(var_set):
        dict_append = dict(zip(k, [feature + '_' + str(i)] * len(k)))
        dict_var.update(dict_append)
    return dict_var
