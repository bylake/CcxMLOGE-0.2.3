
var2woe, iv_get = varf.woeTransData(rawTrainData,train_target,res[1],BinNum,True,BinClass) # 这一步慢
var2woe,dict_filt = varf.filterIV(iv_get,var2woe) # iv值筛选后，并转成了woe的变量
var_vifilter = filterMultiCol(var2woe,thres=thres_vif) # vif 筛选变量

a=var2woe # binclass=0
b=var2woe

cols = set(a.columns).intersection(set(res[1][0]))
a1 = a[list(cols)]

colsb = set(b.columns).intersection(set(res[1][0]))
b1 = b[list(colsb)]

aa1 = a1.apply(lambda i: i.nunique(),axis=0)
bb1 = b1.apply(lambda i: i.nunique(),axis=0)

var_class = pd.concat(map(lambda i: get_iv_woe_class(i, rawTrainData, y, isdump=False, flag=binClass)[1], res[1][0]),
                      axis=1)
