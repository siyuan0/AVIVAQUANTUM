import random
def listshuffle(l):
    l_new = [e for e in l]
    random.shuffle(l_new)
    return l_new
a = [1,2,3,4,5]
b = [a,a,a]
c = [listshuffle(x) for x in b]
print(a)
print(b)
print(c)