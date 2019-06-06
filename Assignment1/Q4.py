A = [(1,2), (4,3), (2,10), (12, 5), (6, 7), (9,11), (15, 4)]

def Sortby1stAscend(A):
    A.sort(key = lambda x:x[0])    
    print(A)

def Sortby2ndAscend(A):
    A.sort(key = lambda x:x[1])    
    print(A)
    
def Sortby1stDescend(A):
    A.sort(key = lambda x:x[0], reverse = True)    
    print(A)
    
def Sortby2ndDescend(A):
    A.sort(key = lambda x:x[1], reverse = True)    
    print(A)
    

Sortby1stAscend(A)
Sortby2ndAscend(A)
Sortby1stDescend(A)
Sortby2ndDescend(A)