import matplotlib.pyplot as plt

# 三条线(1,1) -> (4,2)
# x =|1 2 3| y = |1 1 1|
#    |4 5 6|     |2 2 2|
# 每列绘成一条直线，前面对应
x = [[1,2,3],[4,5,6]]
y = [[1,1,1],[2,2,2]]

# 绘成网格点 x = transpose(y) 且两者中必有行是同一个数或者列是同一个数

plt.plot(x,y,c='r')
plt.show()

x = [1,2,3]
y = [1,2,4]
plt.plot(x,y,c='r')
plt.show()


#==========numpy++++++++
# numpy.ravel() numpy.flatten() 都是将矩阵扁平化，
# 一个拷贝数组中的元素一个引用数组中的元素
