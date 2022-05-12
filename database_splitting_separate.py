if __name__ == '__main__':
    new_str1 = str()
    new_str2 = str()
    new_str3 = str()
    i = 1
    k = 1
    with open('databases/wdbc_split_norm.csv', 'r') as f:
        for line in f:
            if i == 1:
                i += 1
                continue
            label, *data = line.split(',')
            if k == 1:
                new_str1 += label + ',' + ','.join(data)
            if k == 2:
                new_str2 += label + ',' + ','.join(data)
            if k == 3:
                new_str3 += label + ',' + ','.join(data)
            i += 1
            if k == 3:
                k = 1
            else:
                k += 1

print(new_str3)