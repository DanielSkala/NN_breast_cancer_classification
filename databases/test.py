if __name__ == '__main__':
    string = str()
    with open("wdbc.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            string += ','.join(line[1:])
    print(string)