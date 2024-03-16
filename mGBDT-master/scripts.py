import os
import re
import pandas as pd
# 设置目录路径
import shutil


def split_log():
    """
    处理日志，因为日志中有其他制品的结果，将这个只取前304行作为结果，把其他制品的结果去掉
    """
    directory_path = r'./outputs/logs'

    # 遍历目录下的所有文件夹
    for root, dirs, files in os.walk(directory_path):
        for directory in dirs:
            # 找到名为 "None.log" 的文件
            file_path = os.path.join(root, directory, 'Boruta_select\\None.log')
            if os.path.isfile(file_path):
                # 打开文件并读取前152行
                with open(file_path, 'r', encoding="utf-8") as file:
                    lines = file.readlines()[:304]
                # 将前152行写回文件
                with open(file_path, 'w', encoding="utf-8") as file:
                    file.writelines(lines)


def move_file():
    root_move = r"./outputs/2"
    root_moved = r"./outputs/logs"
    dirs = os.listdir(root_move)
    for dir in dirs:
        # if dir == 'EasyClinic' or dir == 'LibEST':
        #     iterdirs = os.listdir(os.path.join(root_move, dir))
        #     for iterdir in iterdirs:
        #         feature_move = os.path.join(root_move, dir, iterdir, "Boruta/Boruta_selected_features.xlsx")
        #         feature_moved = os.path.join(root_moved, dir, iterdir, "Boruta/Boruta_selected_features.xlsx")
        #         if os.path.exists(feature_moved):
        #             os.remove(feature_moved)
        #         if not os.path.exists(feature_move):
        #             print(feature_move)
        #             raise ValueError("没有生成特征!")
        #         shutil.copy(feature_move, feature_moved)
        #     continue
        feature_move = os.path.join(root_move, dir, "Boruta_select/None.log")
        feature_moved = os.path.join(root_moved, dir, "Boruta_select/None.log")
        if os.path.exists(feature_moved):
            os.remove(feature_moved)
        if not os.path.exists(os.path.dirname(feature_moved)):
            os.makedirs(os.path.dirname(feature_moved))
        if not os.path.exists(feature_move):
            print(feature_move)
            raise ValueError("没有生成特征!")
        shutil.copy(feature_move, feature_moved)


# 解析日志
def jie_xi__log(layer_num):
    root = "./outputs/logs"
    # dirs = os.listdir(root)
    dirs = ['CM1_NASA', 'eTOUR', 'GANNT', 'iTrust', 'UC_CC', 'UC_ID', 'UC_TC']
    result_list = []
    for path in dirs:
        # 读取三个日志文件
        # logs = ['None.log', 'Boruta_select/None.log', 'self_select/None.log']
        logs = ['None.log']
        lines_list = []
        artifact_name = path
        print(artifact_name)
        path = os.path.join(root, path)
        # 遍历每个日志文件
        for i, log_file in enumerate(logs):
            filepath = os.path.join(path, log_file)
            # 读取日志文件
            with open(filepath, 'r') as f:
                lines = f.readlines()

            # 初始化结果字典
            results = {
                'f1_max': {'f1': 0.0, 'f2': 0.0, 'precision': None, 'recall': None, 'loss': None, 'epoch': None},
                'f2_max': {'f1': 0.0, 'f2': 0.0, 'precision': None, 'recall': None, 'loss': None, 'epoch': None},
                'last': {'f1': 0.0, 'f2': 0.0, 'precision': None, 'recall': None, 'loss': None, 'epoch': None}
            }

            # 编译正则表达式
            if i == 2:
                pattern = re.compile(
                    r'\[epoch=(\d+)/\d+\]\[test\] loss=(\d+\.\d+), precision=(\d+\.\d+), recall=(\d+\.\d+), f1=(\d+\.\d+), f2=(\d+\.\d+)')
            else:
                pattern = re.compile(
                    r'\[epoch=(\d+)/\d+\]\[test\] ' + artifact_name + '-None loss=(\d+\.\d+), precision=(\d+\.\d+), recall=(\d+\.\d+), f1=(\d+\.\d+), f2=(\d+\.\d+)')

            # 遍历日志文件的每一行
            for line in lines[2:304]:
                # 如果这一行是测试结果
                if '[test]' in line:
                    # 解析测试结果
                    match = pattern.search(line)
                    epoch, loss, precision, recall, f1, f2 = [float(x) for x in match.groups()]
                    # 更新结果字典
                    if f1 > results['f1_max']['f1']:
                        results['f1_max'] = {'epoch': epoch, 'loss': loss, 'precision': precision, 'recall': recall,
                                             'f1': f1, 'f2': f2}
                    if f2 > results['f2_max']['f2']:
                        results['f2_max'] = {'epoch': epoch, 'loss': loss, 'precision': precision, 'recall': recall,
                                             'f1': f1, 'f2': f2}
                    results['last'] = {'epoch': epoch, 'loss': loss, 'precision': precision, 'recall': recall, 'f1': f1,
                                       'f2': f2}

            # 将结果字典转换为DataFrame，并添加后缀到列名
            if i == 0:
                suff = "None"
            elif i == 1:
                suff = "Self"
            else:
                suff = "Boruta"
            df = pd.DataFrame.from_dict(results, orient='index').add_suffix('_' + suff)
            # 将DataFrame添加到列表中
            lines_list.append(df)

        # 使用pd.concat()函数将三个DataFrame拼接成一个
        df_result = pd.concat(lines_list, axis=1)
        # 分割列索引字符串并按照前面的字符串进行分组
        groups = df_result.groupby(df_result.columns.str.split('_').str[0], axis=1)
        # 合并分组后的列
        df_result = pd.concat([group[1] for group in groups], axis=1)
        # 将3*18转化为18*3，先将每一行1*18转化为6*3的，然后拼接
        transform_list = []
        for index, row in df_result.iterrows():
            index_name = []
            for i in range(int(len(row.index) / 1)):
                index_name.append(row.index[i * 1].split("_")[0])
            # for i in range(int(len(row.index) / 3)):
            #     index_name.append(row.index[i * 3].split("_")[0])
            # col_name = [row.index[0].split("_")[1], row.index[1].split("_")[1], row.index[2].split("_")[1]]
            col_name = [row.index[0].split("_")[1]]
            row_df = pd.DataFrame(row.values.reshape(6, 1), index=index_name, columns=col_name)
            row_df = row_df.append(pd.Series("------", index=col_name), ignore_index=True)
            index_name.append("------")
            row_df.index = index_name
            transform_list.append(row_df)
        df_result = pd.concat(transform_list, axis=0)
        result_list.append(df_result)
    # 将结果写入Excel表格
    results = pd.concat(result_list, axis=1)
    results.to_excel(os.path.join("./outputs/performance", 'results' + str(layer_num) + '.xlsx'), index=True)


# split_log()
if __name__ == '__main__':
    jie_xi__log(1)
