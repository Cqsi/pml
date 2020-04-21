def loadData(path):
    x = []
    y = []
    with open(path) as file:
        for line in file:
            # x.append([1, int(line.split()[0])]) makes first column as 1's
            x.append(float(line.split()[0]))
            y.append(float(line.split()[1]))
    
    return [x, y]