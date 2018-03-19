# Python
import csv
import os

err_Wafer_ID_col = 'WAFERID'
err_Field_Center_X_mm = 'FIELDCENTER_X[mm]'
err_Field_Center_Y_mm = 'FIELDCENTER_Y[mm]'
err_Target_Pos_X_mm = 'TARGETPOS_X[mm]'
err_Target_Pos_Y_mm = 'TARGETPOS_Y[mm]'
err_OV_X_nm = 'OV_X[nm]'
err_OV_Y_nm = 'OV_Y[nm]'

err_Sensing_Light = 'SENSINGLIGHT_X[nm]'  # in this case we use only 'x' data ('x' data same as 'y')

matrix = []
Sensing_List = []
input_file_list = []


def add_Sensing_List(light_value):
    flag = 0
    for item in Sensing_List:
        if item == light_value:
            flag = 1

    if flag == 0:
        Sensing_List.append(light_value)


def print_Sensing_List(wafer_id):
    print(' --> Wafer ID : ' + wafer_id)

    for item in Sensing_List:
        print('    + ' + item)



def get_csv_reader(filename, delimiter):
    reader = []
    if not os.path.isfile(filename):
        csvfile = open(filename, "w")
    else:
        csvfile = open(filename, "r")
        reader = csv.DictReader(csvfile, delimiter=delimiter)

    return list(reader)


def check_directory(dctry):
    if not os.path.exists(dctry):
        os.makedirs(dctry)


def remake_wafer_id(wafer_id):
    new_id = ""
    for c in wafer_id:
        if c == '.':
            new_id += ("_")
        else:
            new_id += (c)

    return new_id


def get_current_directory_file_list():
    for root, dirs, files in os.walk('./inputs'):
        for f in files:
            if f.endswith('.csv'):
                input_file_list.append(f)

    print("Input File List")



def read_csv_and_write_output_data(FileName):
    print(' - ' + FileName)

    new_FileName = remake_wafer_id(FileName)
    print(' - new file name : ' + new_FileName)

    csvReader = get_csv_reader('./inputs/' + FileName, ",")
    wafer_id = "NULL"
    for row in csvReader:
        wafer_id = row[err_Wafer_ID_col]
        add_Sensing_List(row[err_Sensing_Light])

    # Wafer Id's dot '.' have to replace character '_'
    new_wafer_id = remake_wafer_id(wafer_id)
    print_Sensing_List(wafer_id)

    for sensing_item in Sensing_List:
        # Create CSV file for Writing
        #output_dir = './outputs/' + new_FileName + '/' + new_wafer_id + '/' + sensing_item + '/'
        output_dir = './outputs/'
        check_directory(output_dir)
        with open(output_dir + '.csv', 'w', newline='') as csvfile:
            # 1. add Wafer ID
            writer = csv.writer(csvfile, delimiter=',')
            #writer.writerow([wafer_id])

            # 2. add datas
            for row in csvReader:
                #print(row)

                if row[err_Sensing_Light] == sensing_item:
                    writer.writerow(
                        [str(float(row[err_Target_Pos_X_mm]) + float(row[err_Field_Center_X_mm]))]
                        + [str(float(row[err_Target_Pos_Y_mm]) + float(row[err_Field_Center_Y_mm]))]
                        + [(row[err_Target_Pos_X_mm])]
                        + [(row[err_Target_Pos_Y_mm])]
                        + [(row[err_OV_X_nm])]
                        + [(row[err_OV_Y_nm])]
                    )


get_current_directory_file_list()
for f_name in input_file_list:
    print(f_name)
    #read_csv_and_write_output_data(f_name)