from ccxMLogE.outputTransform import f_mkdir
from datetime import datetime
def Logis_mkdir(userPath,base,reqId):
    '''

    :param userPath: 用户工作目录
    :return:
    '''

    path = f_mkdir(userPath, 'ccxLogistic')  # 建立文件夹
    project_path = f_mkdir(path,base['programName']) # 建立项目文件夹
    # project_path_sub = f_mkdir(project_path,datetime.now().strftime('%Y%m%d%H%M%S'))
    project_path_sub = f_mkdir(project_path, reqId)
    _ = f_mkdir(project_path_sub, 'pkl')  # 建立pkl文件夹
    _ = f_mkdir(project_path_sub, 'classdict')  # 建立类别字典文件夹
    _ = f_mkdir(project_path_sub, 'ivRes')  # 建立ivTable文件夹
    _ = f_mkdir(project_path_sub, 'model')  # 建立model文件夹
    _ = f_mkdir(project_path_sub,'log') # 建立card文件夹
    return project_path_sub