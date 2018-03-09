import pickle

class DataType():

    def __init__(self):
        self.var_set=0
    def f_VarTypeClassfiy(self,df, cateList):
        '''

        :param df: 数据集
        :param cateList: 用户指定的分类型变量 列表
        :return: 连续型变量的列表 少分类型变量的列表 多分类型变量的列表 需要one-hot处理的变量列表
        # 1 分类型和取值个数小于10的连续型变量  2 多分类型  0 连续型
        '''
        cate_ = df.select_dtypes(include=[object, bool]).columns.tolist()  # 一定是分类型变量 bool为1211新增
        num_ = df.select_dtypes(include=[int, float, 'int64', 'float64']).columns.tolist()  # 连续型变量备选
        # 不在这两个list的变量类型有 时间类型，unit64类型等等
        aa = df.apply(lambda i: i.nunique())
        nunique10ls = aa[aa >= 10].index.tolist()  # 取值个数大于10的变量列表
        nunique15ls = aa[aa >= 15].index.tolist()  # 取值个数大于15的变量列表
        nunique2ls = aa[aa <= 2].index.tolist()  # 取值个数小于2的变量列表，主要用于判断是否需要one-hot

       # 既是字符型变量同时取值个数超过15个
        cate_2 = list(set(cate_) & set(nunique15ls))  # 判断出的多分类，隐藏bug没判断出的多分类，如用户自定义了 set(cate_) | set(cateList)
       # 既是数值型变量，同时取值个数超过10个
        cate_0 = list(set(num_) & set(nunique10ls))  # 连续型变量
        # 多分类变量确认
        # 既是用户指定的分类变量，同时取值个数超过15个
        cate_2_ = list(set(cateList) & set(nunique15ls))  # 用户输入分类型变量 且 个数大于15
        # 取值超过15个的字符型变量或者取值超过15个的用户指定分类变量
        cate_2 = list(set(cate_2) | set(cate_2_))  # 多分类变量
        # 连续型变量
        # 取值超过10个的数值型变量同时不在用户指定的分类变量里
        cate_0 = list(set(cate_0) - set(cateList))
        # 分类型
        # 左侧为字符型变量并扣除掉取值个数超过15个的用户指定分类变量，右侧为数值型变量，并扣除掉取值超过10个的同时不在用户指定的分类变量里的数值型变量
        cate_1 = list(set(cate_) - set(cate_2) | (set(num_) - set(cate_0))) # 左侧为
        # 1127晚 发现一个多分类变量 即在cate_2里 也在 cate_1里 原因是(set(num_) - set(cate_0))带来的
        if (set(cate_2) & set(cate_1)):
            x = set(cate_2) & set(cate_1)
            cate_1 = list(set(cate_1) - x)
        # 二值型的分类变量，需要用one-hot而且是能够直接one-hot的
        one_hot = list(set(cate_1) - set(nunique2ls))
        var_num = cate_0
        var_class = list(set(cate_1).union(set(cate_2)))
        self.var_set = [var_class, var_num]
        # return cate_0, cate_1, cate_2, one_hot
        return self.var_set

    def save_type(self):
        with open('type_data.pkl','wb') as f: pickle.dump(self.var_set,f)

    def load_type(self):
        with open('type_data.pkl', 'rb') as f: DataType = pickle.load(f)
        return DataType
