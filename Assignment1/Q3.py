d = {'Black':'r', 'Hero': 'e', 'Go':'g', 'Clue':'i', 'Mean':'q','Groan':'o','Sin':'p',
'Pint':'u','Tone':'n','Graze':'s','Sea':'t','Plant':'a'}

d2 = {}
for i in d:
    d2[d[i]] = 1

A = []
B = []
C = []
for i in d.keys():
    A.append(i.lower())    
    B.append(i)

for j in range(len(A)):
    flag = 0
    for k in range(len(A[j])):
        if(d2.get(A[j][k],0) == 0):
            flag = 1
            break
    
    if(flag==0):
        C.append(B[j])
        
print(C)        
            