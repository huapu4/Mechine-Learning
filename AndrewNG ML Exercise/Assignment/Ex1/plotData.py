import matplotlib.pyplot as plt


def plot_data(X, y):

    plt.scatter(X, y, c='r', marker='x')  #数据分布图
    plt.xlabel('Profit in $10,000s')
    plt.ylabel('Population of City in 10,000s')
    plt.show()


