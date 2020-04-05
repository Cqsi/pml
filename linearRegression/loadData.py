def loadData(path):
    x = []
    y = []
    with open(path) as file:
        for line in file:
            # x.append([1, int(line.split()[0])]) makes first column as 1's
            x.append(int(line.split()[0]))
            y.append(int(line.split()[1]))
    
    return [x, y]