"""

此脚本主要完成：
1.模型部署的代码生成
2.输入参数的API文档
3.在ccxModelApi脚本中，新增一个接口，等待着模型部署的指令
    把之前的数据重新读取一遍，给出参数的API文档，这样之前的processData类可以复用，不用改了
    里面有模型，有重要变量列表

"""
import simplejson
import pandas as pd
import json

from ccxMLogE.outputTransform import f_getRawcolnames


def f_varApiDesc(df):
    '''
    实现对传入的原始数据的一个统计，统计内容有
    英文变量名 <字段是否允许缺失> 字段能否取缺失值 字段类型
    varName keyIsNull valueIsNull VarType
    <字段是否允许缺失> 这个需要通过重要变量来更新
    :param df:原始数据集
    :return:统计好的DataFrame
    '''
    resDict = {}
    resDict['varName'] = df.columns
    resDict['keyIsNull'] = 'No'
    resDict['valueIsNull'] = df.isnull().sum().values > 0
    resDict['VarType'] = df.dtypes.apply(f_transType)

    return pd.DataFrame(resDict)


def f_updateApiDesc(processData, Apidf):
    '''
    用于更新哪些变量是必须的
    :param processData:
    :param Apidf:
    :return:
    '''
    bst = processData.getbstmodel()  # 取到最优模型
    modelname = processData.getmodelname()  # 取到模型名字
    if modelname == 'ccxboost':
        re = f_getRawcolnames(bst.feature_names, Apidf.varName)  # 取到重要变量，还是反推回原始的输入变量
    elif modelname == 'ccxgbm':
        re = f_getRawcolnames(bst.feature_name(), Apidf.varName)
    elif modelname == 'ccxrf':
        # 随机森林这里 需要去修改一下底层源码了 ccxmodel 里面
        re = f_getRawcolnames(bst[1], Apidf.varName)
    elif modelname == 'ccxscorecard':
        #
        varname = bst.params.index.tolist()
        varname.remove('intercept')
        re = pd.DataFrame({"varname": varname})

    re = re.varname.values

    def f_(x, re):
        if x in re:
            return 'No'
        else:
            return 'Yes'

    def f__(x):
        if x:
            return 'Yes'
        else:
            return 'No'

    Apidf['keyIsNull'] = Apidf.varName.apply(lambda x: f_(x, re))
    Apidf['valueIsNull'] = Apidf.valueIsNull.apply(f__)
    Apidf['fieldLevel'] = 'Level 2'

    dataTop = pd.DataFrame({'fieldLevel': ['Level 1', 'Level 1', 'Level 1'],
     'varName': ['reqIndex', 'reqTime', 'dataJson'],
     'VarType': ['String', 'String', 'List'],
     'keyIsNull': ['No', 'No', 'No'], 'valueIsNull': ['No', 'No', 'No']})

    Apidf = pd.concat([dataTop, Apidf])

    return Apidf[['varName', 'fieldLevel', 'VarType', 'keyIsNull', 'valueIsNull']]


def f_transType(x):
    typeDict = {'int64': 'Numeric', 'float64': 'Numeric', 'object': 'String',
                'int': 'Numeric', 'float': 'Numeric'}
    try:
        return typeDict[str(x)]
    except KeyError:
        return 'String'


# 部署代码的生成，好玩，有挑战

def f_gendeployCode(codefile, ApiName, modelPath, indexName, port):
    '''
    生成模型部署的脚本，然后通过执行此脚本文件，达到模型上线的功能
    :param codefile: 生成的脚本存放路径
    :param ApiName: URL中模型接口的API名称，用于标识一个接口
    :param modelPath: 模型路径，主要指processData对象的路径
    :param indexName: 样本的索引，不强制用户传，如果不想传，给个默认值
    :param port: 端口号，需要用户指定，且不能是已经被使用的
    :return: 返回一个脚本文件路径
    '''
    strDeploy = """
import numpy as np
import pickle
import pandas as pd
import time
from ccxMLogE.predictModel import predictmodel  
from sanic import Sanic
from sanic import response

app = Sanic()


@app.post('/ApiName/{ApiName}')
async def ccxModelApi(request):
    def f_trans_data(data):
        try:
            return pd.DataFrame(data).fillna(np.nan)
        except Exception as e:
            raise ValueError('Data format is not satisfied')

    try:
        st = time.time()
        # 1.解析数据

        Input = request.json
        reqIndex = Input.get('reqIndex')
        reqTime = Input.get('reqTime')
        dataJson = Input.get('dataJson')
        # 2.转换json数据data为DataFrame
        Data = f_trans_data(dataJson)
        # 3.加载模型对象
        with open(r'{modelPath}', 'rb') as f:  # modelPath
            processData = pickle.load(f)
        # 4.预测出结果
        re = predictmodel(processData, Data, '{indexName}')['predictProb'].values.tolist()  # indexName
        # 5.返回结果
        res = {rightres} 
        return response.json(res)
    except Exception as e:
        return response.json({errorres}) 

if __name__ == '__main__':
    app.run(debug=True, port={port}, host='0.0.0.0')""". \
        format(ApiName=ApiName,
               modelPath=modelPath, indexName=indexName, port=port,
               rightres="""{'code': 200, 'reqIndex': reqIndex, 'reqTime': reqTime, 'predictProb': re}""",
               errorres="""{'code': 500,  'resMsg': str(e)}""", )

    with open(codefile, 'wt', encoding='utf-8') as f:
        print(strDeploy, file=f)

    print('代码生成成功')
    return codefile


import random


def f_genPort():
    '''
    随机生成一个5位数，作为端口号
    :return: 5位整数
    '''
    return random.randint(10000, 65535)


import subprocess


def f_checkPort(portNum):
    '''
    检查后台服务器，此端口号是否被占用
    :param portNum: 端口号，5位整数
    :return: 是否被占用的标识 bool {False:被占用，True:未被占用}
    '''
    try:
        subprocess.check_output("netstat -apn | grep {}".format(portNum), shell=True)
        return False
    except subprocess.CalledProcessError:
        # 走这支说明没有被占用
        return True


def f_regenPort():
    '''
    重新生成一个未被占用的端口号
    :return:
    '''
    # 1.随机生成一个端口号
    portNum = f_genPort()
    # 2.检查端口号是否被占用
    flag = f_checkPort(portNum)
    # 3.重新生成端口号，直到生成的端口号满足未被占用的条件
    while True:
        if not flag:
            # 1.随机生成一个端口号
            portNum = f_genPort()
            # 2.检查端口号是否被占用
            flag = f_checkPort(portNum)
        else:
            break
    return portNum


def f_genshellCode(serverPath, shellcodePath):
    '''
    # 生成启动/暂停 模型服务的sh脚本
    :param serverPath: sh脚本和API.py脚本所在的文件夹路径
    :param shellcodePath: 生成的runmodel.sh 脚本的绝对路径
    :return: 路径
    '''

    strshell = """
SERVER={}  # 项目路径
cd $SERVER

case "$1" in

start)
    nohup python deployModelApi.py 1>$SERVER/Modelruninfo.log 2>&1 &
    echo "启动模型服务成功 $!"
    echo $! > $SERVER/server.pid
        ;;

stop)
    kill `cat $SERVER/server.pid`
    echo "停止模型服务成功"
    rm -rf $SERVER/server.pid
    ;;

esac

exit 0""".format(serverPath)

    with open(shellcodePath, 'wt', encoding='utf-8') as f:
        print(strshell, file=f)
    print('运行模型服务的脚本已生成')
    return shellcodePath


def f_runshell(shellPath):
    '''
    使用python 执行shell脚本的函数
    :param shellPath: 即serverpath，sh脚本的文件夹路径;0205修改为fabric的路径
    :return:
    '''
    try:
        import os
        x = subprocess.check_output("fab -f {}/fabricdeploy.py deploy".format(shellPath),
                                    timeout=3, start_new_session=True, shell=True)

        print(x)
        return True
    except Exception as e:
        raise RuntimeError("run shell Error {}".format(str(e)))
        return False


def f_genTestJson(df, n):
    dataJson = df.head(n).to_dict(orient='records')
    res = {'reqIndex': 'index',#请求索引/流水号
           'reqTime': 'time',#请求时间戳
           'dataJson': dataJson
           }

    return simplejson.dumps(res, ensure_ascii=False, ignore_nan=True)


def f_jiexiModelre(reps):
    '''
    解析部署好的模型的测试返回，以此来确认，模型时候部署成功，是否正确返回
    :param reps: 返回内容（有返回，正确或错误 code200/500;或是异常，如接口不存在）
    :return:
    '''
    try:
        _ = json.loads(reps)['code']
        if _ == 200:
            print("模型部署成功，测试结果正常返回")
            return True
        elif _ == 500:
            print("模型部署成功，测试未正常返回")
            return True
    except Exception as e:
        print("模型部署失败")
        return False


def f_ApiDocWriter(path, url, APIdf, testOneJson, r1, testmultiJson, r2):
    writer = pd.ExcelWriter(path)
    durl = pd.DataFrame({"模型接口URL": [url]})
    dTest = pd.DataFrame({"数据说明": ['单条测试', '批量测试'],
                          "测试数据": [testOneJson, testmultiJson],
                          "测试结果": [r1.text, r2.text]
                          })
    durl.to_excel(writer, 'url', index=False)
    APIdf.to_excel(writer, 'param', index=False)
    dTest.to_excel(writer, 'test', index=False)
    writer.save()
    return path

def f_getip():
    # 自动获取部署服务器的ip地址
    import socket
    hostname = socket.gethostname()
    print('fargfregeg', hostname)
    # ip = socket.gethostbyname(hostname)
    ip = '192.168.100.175'
    return ip


def f_genfabricCode(serverPath, shellcodePath):
    '''
    # 生成部署 模型服务的py脚本 后期要实现了远程部署功能
    :param serverPath: py脚本和API.py脚本所在的文件夹路径
    :param shellcodePath: 生成的fabricdeploy.py 脚本的绝对路径
    :return: 路径
    '''

    strshell = """
from fabric.api import *

env.hosts = '127.0.0.1'

def deploy():
    with lcd('{}'):
        local('chmod 755 runmodel.sh')
        local('bash runmodel.sh start && sleep 1')
        print('本地的模型服务启动成功')""".format(serverPath)

    with open(shellcodePath, 'wt', encoding='utf-8') as f:
        print(strshell, file=f)
    print('部署模型服务的fabric脚本已生成')
    return shellcodePath



# def f_subprocess():
#     print('开始启动另一个服务')
#     # os.system('python {}/flask_test.py'.format(versionpath))
#     # os.system('python /root/jupyterhub/CcxMLOGE/Testdeploy/modelDB/rtryyv01/flask_test.py')
#     os.system('python /root/jupyterhub/CcxMLOGE/Testdeploy/modelDB/rtryyv01/deployModelApi.py')
#     print('启动另一个服务的代码已执行')
#     # os.system('python {}/deployModelApi.py'.format(versionpath))
#

# def f_subprocess(versionpath):
#     flag = f_runshell(versionpath)
#     if flag:
#         print('伪启动成功')
#     else:
#         print('启动非常失败,都是坤总的锅，就赖他就好了')



if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\base14_1130.csv')

    f_varApiDesc(df).to_csv(r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\modelAPI_demo.csv', index=False)

    ApiName, modelPath, indexName, port = 'ApiName', r'C:\Users\liyin\Desktop\CcxMLOGE\TestUnit\predict\ccxboost_Xgboost-128-721742.model', 'contract_id', 3333

    codefile = r'C:\Users\liyin\Desktop\CcxMLOGE\code_DF\deployModelApi.py'
    f_gendeployCode(codefile, ApiName, modelPath, indexName, port)

    serverPath = r'C:\Users\liyin\Desktop\CcxMLOGE\code_DF'
    shellcodePath = r'C:\Users\liyin\Desktop\CcxMLOGE\code_DF\runmodel.sh'
    f_genshellCode(serverPath, shellcodePath)

    '''
        Input 的数据结构
        请求数据格式：
        {'reqIndex': '请求索引/流水号',
        'reqTime': '请求时间戳',
        'data': '数据'#[{}]
        }
    '''

    f_genTestJson(df)
    '{"reqIndex": "请求索引/流水号", "reqTime": "请求时间戳", "dataJson": [{"contract_id": 29586, "apply_dts": 297.0, "apply_amount": 2000.0, "apply_month": 3, "login_origin": 1, "mobile_company": 2, "net_age": NaN, "reapply_cnt": 2, "is_night_apply": 0, "is_workday_apply": 1, "is_worktime_apply": 1, "tg_location1": "sdzj", "cid_prov_lvl": NaN, "address_length": 27, "bank_prov_lvl": 2.0, "mobile_segment": 182, "age": 35, "gender": 1, "regaud_diffdays": 2307477.0, "login_diffdays": 73, "mobile_prov_lvl": 2.0, "job_type": 2, "mobile_cardmobile": 1, "addrcity_mobilecity": 1, "addrcity_compcity": 1, "mobilecity_compcity": 1, "cidcity_bankcity": 0, "rela1_lvl": 3.0, "rela2_lvl": 1.0, "rela3_lvl": 2.0, "addrcity_bankcity": 1, "target": 0, "tg_location": 6.0, "mobile_segment_2num": 18}]}'
