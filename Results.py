import oslo as os
#--------------------------------
#TASK 1:
#--------------------------------

#Tests to validate that code is functioning as anticipated
test1 = False
if test1: #Track average height of site i=1 over time in steady-state:
    sys = os.system(0.5,16)
    sys.drive(10000)
    sys.av_height_site_one()
    sys1 = os.system(0.5,32)
    sys1.drive(10000)
    sys1.av_height_site_one()

test2 = False
if test2: #Set p=0 and check that <s>---> L (as in BTW model)
    sys2 = os.system(0,16)
    sys2.BTW_test(500,10000,[4,8,16,32,64])

#--------------------------------
#TASK 2:
#--------------------------------    
        
task2a = False
if task2a:
    sys3 = os.system(0.5,4)
    sys3.total_height(300000,[4,8,16,32]) 

task2c = False
if task2c:
    sys4 = os.system(0.5,4)
    sys4.height_collapse(3000,[4,8,16],100)

task2d = False
if task2d:
    sys5 = os.system(0.5,4)
    a = sys5.numerical_cross_time([4,8,16,32],5) 

task2efg = False
if task2efg:
    sys6 = os.system(0.5,4)
    sys6.av_height_recurrent(10000,[4,8,16],False,False,False) #3rd arg for 2e, 4th arg for 2f, 5th arg for 2g.

#--------------------------------
#TASK 3:
#--------------------------------    
    
task3ab = False
if task3ab:
    sys7 = os.system(0.5,4)
    sys7.avalanche_size_prob(10000,[4,8,16,32],2,True) #False for 3a, True for 3b
    
task3c = False
if task3c:
    sys8 = os.system(0.5,4)
    sys8.moments([4,8,16],10000,True)




