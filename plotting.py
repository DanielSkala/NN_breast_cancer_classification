import matplotlib.pyplot as plt

if __name__ == '__main__':
    LABELS = []
    DATA = []
    with open('databases/wdbc_split.txt', 'r') as f:
        for line in f:
            print(line, end='')
            DATA.append([line.split(',')[0], float(line.split(',')[4])])
    plt.scatter([x[0] for x in DATA], [x[1] for x in DATA])
    plt.show()
