# -*- coding: utf-8 -*-
"""

For loops: Execute a set of statements, 
once for each item in a list, string, tuples, etc. 

"""

#Example 1
"""
a = [1,2,3,4,5]
b = ["a","b","c"]

for i in a:
    print("The current value is ",i)
    
for x in "microscope":
    print (x)
    
"""

#Example 2 
""" 
microscopes = ["confocal", "widefield", "fluorescence"]

for x in microscopes:
    print(x)
    if x == "widefield":
        break 
"""
 
#Example 3
for i in range(10):
    print(i)    
    
for i in range(20, 60, 2):
    print(i)
    
for num in range(0, 20):
    if num%2 ==0:
        print("%d is an even number" %(num))
    else:
        print("%d is an odd number" %(num))
    

    
    