import numpy as np
import csv

def getData(file_dir):
    f = open(file_dir,'r',encoding='utf-8')
    reader = csv.reader(f)
    idx_line = 0
    datas = []
    for line in reader:
        if idx_line > 0:
            idx = 0
            for num in line:
                num = int(num)
                line[idx] = num
                idx = idx + 1
            #print(line)
            datas.append(line)
        idx_line = idx_line + 1
    f.close()

    num_of_data = idx_line
    #print(num_of_data)
    idx = 0
    input_datas = []
    for data in datas:
        temp_datas = []
        if idx + 7 > num_of_data-1:
            break
        for i in range(7):
            #print(idx+i)
            for num in datas[idx+i]:
                temp_datas.append(num)
            #temp_datas.append(datas[idx+i])
        #print("------------------")
        #print(temp_datas)
        input_datas.append(temp_datas)
        idx += 1

    #for data in input_datas:
        #print(data)

    return input_datas

def getOutputData(file_dir):
    f = open(file_dir, 'r', encoding='utf-8')
    reader = csv.reader(f)
    idx_line = 0
    datas = []
    for line in reader:
        if idx_line > 7:
            idx = 0
            for num in line:
                num = int(num)
                line[idx] = num
                idx = idx + 1
            #print(line)
            datas.append(line)
        idx_line = idx_line + 1
    f.close()

    output_size = 45
    output_datas = []
    for data in datas:
        temp_data = np.zeros(45)
        for num in data:
            #print(num)
            temp_data[(num)-1] = 1
        output_datas.append(temp_data)
        #print(temp_data)
    #data = np.zeros((45))
    #print(data)
    return output_datas
