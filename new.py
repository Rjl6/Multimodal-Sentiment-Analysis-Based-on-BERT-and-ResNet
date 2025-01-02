lis = [1.1,2.21,3.1]

with open('a.txt','w')as f:
    for num in lis:
        print(num,' ')
        f.write(str(num))
        f.write(' ')