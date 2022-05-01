from scipy import stats
import pandas as pd



if __name__ == '__main__':

    df= pd.read_csv("databases/wdbc_split.txt")
    
    new_str = str()
    with open('databases/wdbc_split.txt', 'r') as f:
        for line in f:
            id, label, *data = line.split(',')
            data1 = data[0:10]
            data2 = data[10:20]
            data3 = data[20:30]
            new_str += label + ',' + ','.join(data1) + '\n'
            new_str += label + ',' + ','.join(data2) + '\n'
            new_str += label + ',' + ','.join(data3)

    print(new_str, end='')