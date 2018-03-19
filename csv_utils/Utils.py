import csv
import os
from collections import defaultdict
from config import ExtractConfig as EC
class Utils:
    def __init__(self):
        self.init_input_files()
        self.init_target_files()

    def init_input_files(self):
        self.input_dir = './inputs'
        self.input_file_list = []
        self.input_col_list = EC.extract_list
        #print('init input files...')
        for root, dirs, files in os.walk(self.input_dir):
            for f in files:
                #print(f)
                if f.endswith('.csv'):
                    self.input_file_list.append(self.input_dir + "/" + f)

    def init_target_files(self):
        self.target_file_dir = './target'
        self.target_file_list = []
        self.target_col_list = EC.target_col_list
        #print('init target files...')
        for root, dirs, files in os.walk(self.target_file_dir):
            for f in files:
                #print(f)
                if f.endswith('.csv'):
                    self.target_file_list.append(self.target_file_dir + "/" + f)

    def init_output_files(self,title):
        #print('init ' + title + ' output files...')
        self.output_dir = './outputs/' + title + '/'
        self.output_file_name = title + '_result.csv'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @staticmethod
    def get_csv_reader(filename, delimiter):
        reader = []
        if not os.path.isfile(filename):
            csvfile = open(filename, "w")
        else:
            csvfile = open(filename, "r")
            reader = csv.DictReader(csvfile, delimiter=delimiter)
        return list(reader)

    def print_input_files(self):
        self.print_csv_files(self.input_file_list,self.input_col_list)

    def print_target_files(self):
        self.print_csv_files(self.target_file_list,self.target_col_list)

    def print_csv_files(self,file_list,col_list):
        for f_name in file_list:
            print(f_name)
            csv_reader = self.get_csv_reader( f_name,",")
            for row in csv_reader:
                #print(row)
                for extract_item in col_list:
                    print(extract_item , row[extract_item])
                #return

    def merge_all_input_files(self):
        # init output directory and file name
        self.init_output_files('merge_all')
        # in python 2.7 -> with open(self.output_dir + self.output_file_name, 'wb') as csvfile:
        with open(self.output_dir + self.output_file_name , 'w', newline='') as csvfile:
            # get writer
            writer = csv.writer(csvfile, delimiter=',')
            # set List name on top
            writer.writerow(self.input_col_list)
            # get all files in ./inputs/
            for f_name in self.input_file_list:
                # get csv reader
                csv_reader = self.get_csv_reader(f_name, ",")
                for row in csv_reader:
                    datas = []
                    # get selected index from 'extract config class'
                    for extract_item in self.input_col_list:
                        # accumulate them
                        datas.append(row[extract_item])
                    # write them one row
                    writer.writerow(datas)
                # repeat each files and accumulate in one file

    def select_target_pos(self):
        target_count = 0
        print('target list')
        print('--------------------------------------------------')
        for f_name in self.target_file_list:
            csv_reader = self.get_csv_reader(f_name,",")
            self.target_file_data_temp = []
            for row in csv_reader:
                data = []
                for extract_item in self.target_col_list:
                    data.append(row[extract_item])
                    print('%10s' % row[extract_item],)

                self.target_file_data_temp.append(data)
                print()
                target_count += 1
        print('--------------------------------------------------')

        self.selected_target_list = []
        while True:
            try:
                idx = int(input('select target ID (stop -> -1): '))
            except:
                print('(error) target id must be 1 ~ ', target_count)
                continue

            if idx == -1:
                break
            if idx > 0 and idx <= target_count:
                self.selected_target_list.append(idx)
                print('selected targets:',)
                print(self.selected_target_list)
            else:
                print('(error) target id must be 1 ~ ' , target_count)

        self.target_file_data = []
        for selected_idx in self.selected_target_list:
            self.target_file_data.append(self.target_file_data_temp[selected_idx])

        self.target_dict = defaultdict(lambda: 0)
        for target in self.target_file_data:
            self.target_dict[(target[1],target[2])] = 1

        print('\nselected data')
        for data in self.target_file_data:
            print(data)

    def check_data_pos(self,data):
        pos_x = self.input_col_list[8] # TARGETPOS_X[mm]
        pos_y = self.input_col_list[9] # TARGETPOS_Y[mm]
        for check_data in self.target_file_data:
            #if check_data[1] == row[pos_x] and check_data[2] == row[pos_y]:
            if self.target_dict[(data[pos_x],data[pos_y])] == 1:
                return True
        return False

    def check_data_col_row(self,data):
        col = self.input_col_list[4] # COL
        row = self.input_col_list[5] # ROW
        if self.target_col_row_dict[(col,row)] == 1:
            return True
        return False

    def select_target_col_row(self):
        self.target_col_row_dict = defaultdict(lambda : 0)
        self.target_col_row_data = []
        while True:
            print('select COL and ROW (stop -> 999)')
            try:
                col = int(input('COL: '))
                if col == 999 :
                    break
                row = int(input('ROW: '))
                if row == 999 :
                    break

                self.target_col_row_data.append((col,row))
                print('selected col&row: ',)
                print(self.target_col_row_data)
                self.target_col_row_dict[(col,row)] = 1
            except:
                print('\n(error) select only integer')
                continue

        print('\nselected col&row\n')
        for i in self.target_col_row_data:
            print(self.target_col_row_data[i])

    def merge_selected_input_files(self):
        # select target data
        self.select_target_pos()
        self.select_target_col_row()

        # init output directory and file name
        self.init_output_files('merge_selected')
        # in python 2.7 -> with open(self.output_dir + self.output_file_name, 'wb') as csvfile:
        with open(self.output_dir + self.output_file_name, 'w', newline='') as csvfile:
            # get writer
            writer = csv.writer(csvfile, delimiter=',')
            # set List name on top
            writer.writerow(self.input_col_list)
            # get all files in ./inputs/
            for f_name in self.input_file_list:
                # get csv reader
                csv_reader = self.get_csv_reader(f_name, ",")
                for row in csv_reader:
                    datas = []
                    # is this target row?
                    if self.check_data_pos(row) and self.check_data_col_row(row):
                        # get selected index from 'extract config class'
                        for extract_item in self.input_col_list:
                            # accumulate them
                            datas.append(row[extract_item])
                        # write them one row
                        writer.writerow(datas)
                        # repeat each files and accumulate in one file

    def run(self):
        while True:
            try:
                print('-------------------------------')
                print('select mode(enter only integer)')
                print(' 1 : merge all input files')
                print(' 2 : merge selected input files')
                print('-1 : exit program')
                print('-------------------------------')
                idx = int(input('mode: '))
            except:
                print('\n(error) mode must be integer\n')
                continue

            if idx == -1:
                print('\nbye bye\n')
                break
            if idx == 1:
                self.merge_all_input_files()
                print('\nmerge all input files finish!\n')
            elif idx == 2:
                self.merge_selected_input_files()
                print('\nmerge selected input files finish!\n')
            else:
                print('\n(error) mode must be integer\n')

reader = Utils()
reader.run()
