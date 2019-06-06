
def numofoccure(A): 
      
    d = {} 
    for i in A: 
        d[i] = d.get(i,0) +1
    print(d) 

A = [('a','s','d'), (1,2,6,8), ('a','s','d'),  
               ('t', 'u'), (3, 'q')]  
  
numofoccure(A) 