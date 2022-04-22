if __name__ == '__main__':
    new_str = str()
    with open('databases/wdbc.txt', 'r') as f:
        for line in f:
            id, label, *data = line.split(',')
            data1 = data[0:10]
            data2 = data[10:20]
            data3 = data[20:30]
            new_str += label + ',' + ','.join(data1) + '\n'
            new_str += label + ',' + ','.join(data2) + '\n'
            new_str += label + ',' + ','.join(data3)

    print(new_str, end='')