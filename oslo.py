import random 
import matplotlib
#Set matplotlib parameters for nice plots
matplotlib.rcParams.update({'axes.titlesize':45,'axes.labelsize':30,'xtick.labelsize':25,'ytick.labelsize':25})
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from scipy.optimize import curve_fit
from scipy import stats
from logbin2018new import logbin

class system:
    """
    System is a class which "contains" an entire Oslo model system.
    An object of this class is a system, of input size L and with threshold slope re-assingment probability p
    Class contains all methods used in this project (data collapes, scaling behaviours etc.)
    See Results.py for easy use of this code.
    """
    
    def __init__(self,p,L):
        
        self.p = p  #probability of selecting height = 1
        self.L = L #system size
        self.z = [0]*self.L #iniate system in empty (and therefore trivially stable) configuration
        self.z_prev = [0]*self.L #keep track of system one step previously 
        self.zth = [self.assign_slope() for i in range(self.L)] #assign random threshold slopes 
        self.h = [0]*(self.L+1) #list of current heights 
        self.number_of_grains = 0 #number of grains added to system
        self.dropped = 0 #number of grains dropped during current drive phase
        self.t_c = 0        
        self.steady_state = False #boolean variable = True once system in steady state        
        self.N_relaxations = 0 #number of relaxations at each check phase
        self.s = 0 #total avalanche size for each drive phase
        self.tracker = [] #track height of site i=1 for model testing
        self.avalanche_sizes = [] #track all avalanche sizes over time        
        self.h_t = [] #height of avalanche over time
        self.toppled_i1 = 0 #number of topplings from first pile after each iteration        
        self.average_slope = 0 #the average slope at the step BEFORE adding one grain induces a drop
        self.slope_error = 0 #Standard deviation in average slope measurement ^
        self.stop = 0 #allows us to speed things up by not running beyond required point (e.g. steady state)
        self.cross_time_new = 0 #This is the time when ANY number of grains first leave the system.                
        self.stop_at_exact_cross = False

    def assign_slope(self):
        """ Function returns 1 or 2, with probabilities p and (1-p) respectively"""
        r = random.random()
        if r<=self.p:
            return 1
        else:
            return 2
        
    def oslo_step(self):
        """"Performs one step of the oslo process, from grain addition,through to complete, system-wide stability"""
        self.z_prev = self.z
        self.dropped = 0 
        self.z[0] +=1 #add a grain at first site.
        self.h[0] += 1
        self.s = 0
        restart = True #Only changes value once all possible relaxations are made.
        while restart == True: 
            self.N_relaxations = 0  #count number of relaxations 
            for i in range(self.L):
                if self.z[i] > self.zth[i]:
                    if i == 0:
                        self.z[i] -= 2
                        self.z[i+1] += 1
                        self.toppled_i1 +=1 
                    elif i == (self.L)-1:
                        self.z[i] -= 1
                        self.z[i-1] +=1
                        self.dropped +=1 
                    else:
                        self.z[i] -= 2
                        self.z[i+1] +=1
                        self.z[i-1] +=1
                    self.zth[i] = self.assign_slope()    #randomly assign new slope to relaxed site
                    self.h[i] -= 1                     #keep track of heights
                    self.h[i+1] += 1                    #keep track of heights
                    self.N_relaxations +=1                   #keep track of number of relaxations
                    self.s +=1                  #keep track of current avalanche size
            #Keep iterating the process until ALL sites are relaxed and stable
            if self.N_relaxations == 0:
                restart = False
        
        self.avalanche_sizes.append(self.s)
        self.h_t.append(self.number_of_grains - self.toppled_i1) #height at time t = number of grains - number toppled from site i=1
        if self.stop_at_exact_cross == True: #If we are searching for cross over time t_c (required in some of below methods)
            if self.dropped != 0:
                if self.steady_state == False:
                    self.cross_time_new = self.number_of_grains - 1 #t_c is time before adding one grain causes one grain to leave for first time
                    self.average_slope = np.mean(self.z_prev) #calculate average slope at this point
                    self.slope_error = np.std(self.z_prev)
                self.steady_state = True #System is now in steady-state
                
        if self.dropped == 1: #Alternative definiton of steady-state is when ONLY one grain leaves: this will happen >>>t_c on average
            if self.steady_state == False:
                self.t_c = self.number_of_grains -1 
                self.average_slope = np.mean(self.z_prev) 
                self.slope_error = np.std(self.z_prev)
            self.steady_state = True
        
        #Track height at site i=1 once system is in steady-state for purposes of TESTING
        if self.steady_state == True:
            self.tracker.append(self.h[0])
                       
    def drive(self,num):
        """Drives the oslo model 'num' number of times. Ie. adds 'num' grains to system"""
        if self.stop == True: #If the method only needs to add until steady-state (stops at t_c for efficiency)
            while self.steady_state != True:
                    self.number_of_grains +=1
                    self.oslo_step()
        else: #otherwise
            for i in range(num):
                self.number_of_grains +=1
                self.oslo_step()

    def av_height_site_one(self):
        """return average height of site i=1 over time, once system has reached steady state"""
        results = float(sum(self.tracker))/float(len(self.tracker)), np.std(self.tracker)
        print "Average Height of site i=1 over time, once system in steady state (L=16): ", results[0]," +/- ", results[1] 

    def plot_sizes(self):
        """Plot of avalanche size against 'time' - clock counting number of grains"""        
        plt.figure()
        plt.bar(range(len(self.avalanche_sizes)),[float(i)/float(max(self.avalanche_sizes)) for i in self.avalanche_sizes])
        plt.xlabel("t")
        plt.ylabel("s/smax")
        plt.show()
        
    def BTW_test(self,W,num_grains,sizes):
        """Perform BTW test by setting p=0, for Task 1
        W = window in moving average
        num_grains = number of grains to be added
        sizes = list of system sizes to be trialled"""
        storage = []
        for size in sizes:
            self.reset(0,size) #set system size = 0
            self.drive(num_grains)
            storage.append(self.avalanche_sizes)
        temp_av = []         #Perform moving average to smoooth out the data
        for group in storage:
            num_grains = len(group)
            collect = []
            for i in range(int(num_grains)):    #Algorithm described in Data Smoothing Section of Report
                if i < W+1:
                    window = i*2
                elif i == (num_grains-1):
                    window = 0
                elif i > (num_grains-1-W):
                    window = ((num_grains-1)-i)*2
                else:
                    window = (2*W)+1        
                summ = 0
                if i !=0:
                    if i == num_grains-1:
                        summ = group[i]
                    else:
                        for j in range(i-(window/2),i+(window/2)):
                            summ += group[j]
                if window != 0:   
                    summ = float(summ)/float(window)
                collect.append(summ)
            temp_av.append(collect)
        
        plt.figure() #Plot result of BTW test - should show heights tending to L in steady-state
        for i in temp_av:
            plt.plot(range(len(i)),i,label = "L = %s"%sizes[temp_av.index(i)])
        for i in sizes:
            plt.plot(range(len(storage[-1])),[i]*len(storage[-1]),'--')
        plt.xlabel("t")
        plt.ylabel("s")
        plt.legend(fontsize=20)
        plt.show()
        
        print "In BTW model, we expect <s>----->L in the steady-state. By setting p=0 in the Oslo model, the set-up reduces to this simpler case. Therefore we expect s to ---> constant ~L over time (as observed in plotted graph)"
        
             
    def total_height(self,num_grains,sizes):
        """Plot height h(t;L) vs t for multiple L sizes is list of system sizes to be trialled
        num_grains = number of grains to be added
        sizes = list of system sizes to be trialled"""
        storage = []
        for sys_size in sizes: #Drive the system for each L
            self.reset(self.p,sys_size)
            self.drive(num_grains)
            storage.append((self.h_t,sys_size))
            print "complete for size: ", sys_size
        plt.figure()
        for data in storage:
            plt.plot(range(len(data[0])),data[0],label="system size: %s"%data[1],linewidth=1)
        plt.xlabel("t")
        plt.ylabel("h(t;L)")
        leg = plt.legend(loc=2,markerscale=25., scatterpoints=1, fontsize=20)
        leg.get_frame().set_alpha(0.5)
        plt.show()
        print "Task 2a: Plot height over time for various L (see figure)."
            
    def height_collapse(self,num_grains,sizes,W):
        """Plot height h(t;L) vs t for multiple L, but with re-scaled variables to get data collapse
        num_grains = number of grains to be added
        sizes = list of system sizes to be trialled
        W = moving average window size"""
        storage = []
        for sys_size in sizes: #Drive the system
            self.reset(self.p,sys_size)
            self.drive(num_grains)
            storage.append((self.h_t,sys_size))
        
        temp_av = [] #Perform moving average smoothing
        for group,size in storage:
            collect = []
            for i in range(int(num_grains)):    
                if i < W+1:
                    window = i*2
                elif i == (num_grains-1):
                    window = 0
                elif i > (num_grains-1-W):
                    window = ((num_grains-1)-i)*2
                else:
                    window = (2*W)+1        
                summ = 0
                if i !=0:
                    if i == num_grains-1:
                        summ = group[i]
                    else:
                        for j in range(i-(window/2),i+(window/2)):
                            summ += group[j]
                if window != 0:   
                    summ = float(summ)/float(window)
                collect.append(summ)
            temp_av.append((collect,size))
            
        #Plot new data with temporal averaging over window [t-W,t+W]
        plt.figure()
        for data,size in temp_av:
            plt.scatter(range(len(data)),data,label="system size: %s"%size,marker='x',s=0.05)
        plt.xlabel("t")
        plt.ylabel(r'$\tilde{h}(t;L)$')
        plt.title("Total Height of Pile - data smoothed using moving average")
        leg = plt.legend(loc=2,markerscale=25., scatterpoints=1, fontsize=12)
        leg.get_frame().set_alpha(1)        
        plt.show()
        
        #Plot re-scaled y axis
        plt.figure()
        for data,size in temp_av:
            plt.plot(range(len(data)),[float(j)/float(size) for j in data],label="system size: %s"%size)
        plt.xlabel("t")
        plt.ylabel(r'$L^{-1}\tilde{h}(t;L)$')
        plt.legend(fontsize=20)
        plt.show()
        
        #Plot re-scaled t axis
        plt.figure()
        for data,size in temp_av:
            plt.plot([float(i)/(float(size)**2) for i in range(len(data))],[float(j)/float(size) for j in data],label="system size: %s"%size,)
        plt.xlabel(r'$t/L^{2}$')
        plt.ylabel(r'$L^{-1}\tilde{h}(t;L)$')
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=20)
        plt.show()
        
        #Full Data Collapse on Linear Sacles
        plt.figure()
        for data,size in temp_av:
            plt.plot([float(i)/(float(size)**2) for i in range(len(data))],[float(j)/float(size) for j in data],label="system size: %s"%size)
        plt.xlabel(r'$t/L^{2}$')
        plt.ylabel(r'$L^{-1}\tilde{h}(t;L)$')
        plt.legend(fontsize=20)
        plt.show()
        
        #Plot whole thing over smaller t range
        plt.figure()
        for data,size in temp_av:
            if size > 1:
                cropped = [] #only fit below exponential to scaling function at SMALL ARGUMENT (ie. <1.0)
                for i in range(len(data)):
                    if (float(i))/(float(size)**2) <=1.0:
                        cropped.append(data[i])
                x = [float(i)/(float(size)**2) for i in range(len(cropped))]
                y = [float(j)/float(size) for j in cropped]
                plt.plot(x,y,linewidth=1)
                fitting = curve_fit(lambda t,a,b: a*(t**b),x,y,p0=[2,0.5])
                a = fitting[0][0]
                b = fitting[0][1]
                #errs = fitting[1]
                #print "Data for low argument follows y= a*(x^b)", a,b
                #print "Error in Exponential Fit: ", np.sqrt(np.diag(errs))[1]
                scale = [10**-6,10**-5,10**-4,10**-3,10**-2,10**-1,0.2,0.3,0.4,0.5,1.0] #so that log plotting works
                plt.plot(scale,[a*(i**b) for i in scale],'--',label='L={0},b={1:.3f}'.format(size,b))
        plt.xlabel(r'$t/L^{2}$')
        plt.ylabel(r'$L^{-1}\tilde{h}(t;L)$')
        plt.xlim([0,1])
        leg = plt.legend(loc=4,markerscale=25., scatterpoints=1, fontsize=20)
        leg.get_frame().set_alpha(1)
        plt.show()
        print """Task 2c: Generate Data Collapse for height. 
        The figures produced are as follows:
            Height against time (smoothed with moving average)
            Vertical data collapse
            Full data collapse (log-log scale)
            Full data collapse (linear scale)
            Small argument FSS scaling function"""
        
    def numerical_cross_time(self,sizes,repetitions):
        """Code to numerical test hypothesis about numerical cross over time in Task 2d
        sizes = list of sizes to be trialled.
        repetitions = number of times systems run to take average over.
        """
        vals = []
        slopes = []
        for size in sizes:
            results = []
            slopes_count = []
            for i in range(repetitions): #Drive the system
                self.reset(self.p,size)
                self.stop_at_exact_cross = True
                self.stop = True #allow program to STOP running once system reaches steady state - saves time
                self.drive(10000000) 
                results.append(self.cross_time_new)
                slopes_count.append(self.average_slope)
            vals.append((results,size))
            slopes.append((slopes_count,size))
            print "done for size: ", size
        slopes_calc = []
        vals_cal = []
        errors = []
        errors_t = []
        errs_to_plot = [] 
        for i,j in slopes: #calculate a mean, peak of histograms for purposes of plotting numerical vs. theoretical
            histogram = np.histogram(i)
            index = np.where(histogram[0] == max(histogram[0]))[0][0]
            peak = (float(histogram[1][index+1]+float(histogram[1][index])))/2.0
            mean = np.mean(i)
            #print "System Size: ",j, ", Mean Slope: ", mean, " Error in Mean: +/- ", np.std(i)
            slopes_calc.append((peak,mean,j,Counter(i).keys()[0])) #add peak, mean, size and Most common value
            errors.append((float(np.std(i))/float(mean))*100)
        for i,j in vals:
            histogram = np.histogram(i)
            index = np.where(histogram[0] == max(histogram[0]))[0][0]
            peak = (float(histogram[1][index+1]+float(histogram[1][index])))/2.0
            mean = np.mean(i)
            #print "System Size: ",j, ", Mean Cross Time: ", mean, " Error in Mean: +/- ", np.std(i)
            vals_cal.append((peak,mean,j,Counter(i).keys()[0]))
            errors_t.append((float(np.std(i))/float(mean))*100)
            errs_to_plot.append(np.std(i)*10)
            
        plt.figure() #Plot Errors in means
        plt.plot(sizes,errors,marker='x',label='Error In <z>')
        plt.plot(sizes,errors_t,marker='x', label = "Error in <t_c>")
        plt.xlabel("L")
        plt.ylabel("% Error in Mean")
        plt.legend(fontsize=20)
        plt.show()
        
        
        #Theoretical result using corrections to scaling 
        def theoretical(L,slope):
            return (float(slope)/2.0)*float(L)*(float(L)+1.0)

        #Plot theoretical formulation^ against numerical data.
        plt.figure()
        plt.errorbar([j[2] for j in vals_cal],[j[1] for j in vals_cal],yerr= errs_to_plot,fmt='o',label="<t_c> +/- 10sigma")
        plt.plot([i[1] for i in vals],[theoretical(i[1],slopes_calc[vals.index(i)][1]) for i in vals],'--',label="Theoretical Prediction") #Using average z
        plt.xlim([0,max([j[1] for j in vals])+10])
        plt.legend(fontsize=20)
        plt.xlabel("L")
        plt.ylabel(r"$t_{c}$")
        plt.show()
        
        #Obtain error in numerical data with respect to the theory
        x = []
        for i in range(len(vals_cal)):
            x.append(((vals_cal[i][1] - theoretical(vals[i][1],slopes_calc[i][1]))**2)/(theoretical(vals[i][1],slopes_calc[i][1])**2)*100)
        plt.figure()
        plt.plot([i[1] for i in vals],x,marker='x')
        plt.xlabel("L")
        plt.ylabel("%Error")
        plt.show()
        
        print """Task 2d: Check that theory agrees with numerical data for cross over time. Figures produced are:
            Errors in means of <z> and <t_c> - to check if mean is a good measure.
            Theory vs. numerical data for cross over time
            Agreement of theory vs data.
        """

    def av_height_recurrent(self,T,sizes,height,sd,prob):
        """Perform Tasks 2e,2f,2g
        T = time period over which averaging done
        sizes = list of system sizes to be trialled
        height = Boolean: True: do Task 2e, False: don't
        sd = Boolean: True: do Task 2f, False: don't
        prob = Boolean: True: do Task 2g, False: don't
        """
        if prob == True:
            sd = True
        results = []
        results_squared = []
        results_prob = []
        slopes_count = []
        slopes_error = []
        for size in sizes:
            #buffer value is amount t0 is greater than t_c, set to be t_c  (ie t0 = tc+tc (therefore t0 >> t_c))
            #thus ensuring that we are definitely in the recurrent configurations part
            self.reset(self.p,size)
            self.stop = True
            self.drive(100000)
            self.stop = False
            self.drive(T+(2*self.t_c)+1)        
            t0 = int(2*self.t_c)
            values = self.h_t[t0: int(t0+T)] #take all times after t0 until t0+T            
            if sd==True:
                values_sq = [i**2 for i in values]
                results_squared.append(np.mean(values_sq))
            results.append(np.mean(values))            
            counted = Counter(values) #count occurences of each height
            total = len(values)
            results_prob.append((size,total,counted))           
            slopes_count.append(self.average_slope)
            slopes_error.append(self.slope_error)
            print "done for size: ", size

        if height == True:
            #Plot average height/L on log/log plot
            plt.figure()
            plt.xlabel("L")
            plt.ylabel("<h>/L")
            plt.plot(sizes,[float(i)/float(sizes[results.index(i)]) for i in results],marker='x',color='red',label='<h>/L')
            #perform least-squares fitting using scipy just as a trial run.            
            fitting = curve_fit(lambda t,a,b,c: a+(1-b*(t**(-c))),sizes,[float(i)/float(sizes[results.index(i)]) for i in results],p0=[2,0.5,1])
            a = fitting[0][0]
            b = fitting[0][1]
            c = fitting[0][2]            
            plt.plot(np.linspace(min(sizes),max(sizes),100),[a+(1-b*(i**(-c))) for i in np.linspace(min(sizes),max(sizes),100)],'--',label='Exponential Fit, ~%.2f'%-c)
            plt.legend(fontsize=20)
            plt.show()
            a0 = np.linspace(0,2,1000) #dummy a_0 value to be tested for corrections to scaling
            values = []            
            plt.figure()
            errors = []
            for a in a0:    
                if a > max([float(i)/float(sizes[results.index(i)]) for i in results]): #check for negative values in log
                    y = [np.log(a-(float(i)/float(sizes[results.index(i)]))) for i in results]
                    x = [np.log(i) for i in sizes]
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                    values.append((a,r_value,slope))
                    plt.plot(x,y,label="a0=%s"%a)
                    errors.append((a,r_value**2))
            plt.xlabel("Log(L)")
            plt.ylabel("Log(a0 - <h>/L)")
            plt.legend(fontsize=20)
            plt.show()
            
            #Find best a_0 value as the one with the hgihest R^2 value
            plt.figure()
            plt.plot([i[0] for i in errors], [i[1] for i in errors])
            best_val = errors[[i[1] for i in errors].index(max([i[1] for i in errors]))][0]
            plt.plot([best_val]*100,np.linspace(0,errors[[i[1] for i in errors].index(max([i[1] for i in errors]))][1],100),'--')
            plt.scatter(errors[[i[1] for i in errors].index(max([i[1] for i in errors]))][0],errors[[i[1] for i in errors].index(max([i[1] for i in errors]))][1],marker='x',color='red',label="Best Estimate = %s"%best_val)
            plt.xlabel("a0 estimate")
            plt.ylabel(r"$R^{2} value$")
            plt.legend(fontsize=20)
            plt.show()
            print "Task 2e: Corrections to scaling in height. First figure is seeing if there are any corrections (if its not constant there are!). Second figure shows effect of testing a range of a_0 values. Third figure is R^2 test to obtain the 'best' a_0 value"""
            
        if sd == True:
            final_results = []
            for i in range(len(sizes)):
                final_results.append(np.sqrt(results_squared[i]-(results[i]**2)))
            #Plot scaling behaviour of sigma
            plt.figure()
            plt.scatter(sizes,final_results,marker='x',label='data points')
            fitting = curve_fit(lambda t,a,b: a*(t**b),sizes, final_results,p0=[1,0.5]) #exponential curve fitting to obtain scaling behaviour
            a = fitting[0][0]
            b = fitting[0][1]
            plt.plot(np.linspace(min(sizes),max(sizes),100),[a*(i**b) for i in np.linspace(min(sizes),max(sizes),100)],'--',label='exponential fit, exponent = %.2f'%b)
            plt.xlabel("L")
            plt.ylabel(r"$\sigma_{h}$")
            plt.legend(fontsize=20)
            plt.show()
            
            plt.figure() #Check behaviour of <z> as L goes to ~inf
            plt.scatter(sizes,slopes_count,marker='x')
            plt.plot(sizes,slopes_count,label="<z> data")
            plt.plot(sizes,[1.738]*len(sizes),'--',label="a_0")
            plt.xlabel("L")
            plt.ylabel("<z>")
            plt.legend(fontsize=20)
            plt.show()
            
            plt.figure() #Behaviour of error in <z> as L goes to ~inf
            plt.scatter(sizes,slopes_error,marker='x')
            plt.plot(sizes,slopes_error)
            plt.xlabel("L")
            plt.ylabel(r"$\sigma_{<z>}$")
            plt.show()
            print "Task 2f: Error in h. First graph is scaling of sigma with L. Second and third graphs are rough guesses at behaviour of <z> and its associated error as L--->> infinity."""
     
        if prob == True:            
            plt.figure() #Plot P(h;L) for multiple system sizes
            for i in results_prob:
                a = [float(i[2][j])/float(i[1]) for j in i[2].keys()]
                b = i[2].keys()
                b,a = zip(*sorted(zip(b, a)))
                plt.plot(b,a,label = "System Size %s" %i[0])
            plt.xlabel("h")
            plt.ylabel("P(h;L)")
            plt.legend(fontsize=20)
            plt.show()

            plt.figure()
            for i in results_prob: #Data Collapse
                a = [float(i[2][j])/float(i[1]) for j in i[2].keys()] #divide by N to get probability
                b = i[2].keys()
                b,a = zip(*sorted(zip(b, a)))                
                plt.plot([(float(j)-float(results[results_prob.index(i)]))/float(final_results[results_prob.index(i)]) for j in b],[float(j)*float(final_results[results_prob.index(i)]) for j in a],label = "System Size %s" %i[0])
            ranger = np.linspace(min([(float(j)-float(results[results_prob.index(i)]))/float(final_results[results_prob.index(i)]) for j in b]),max([(float(j)-float(results[results_prob.index(i)]))/float(final_results[results_prob.index(i)]) for j in b]),100)            
            plt.plot(ranger,[(1.0/((2*np.pi)**0.5))*(np.exp((-1.0/2.0)*(i**2.0))) for i in ranger],'--',label="FSS Ansatz")            
            plt.xlabel("h")
            plt.ylabel("P(h;L)")
            plt.legend(fontsize=20)
            plt.show()
            
            print """Task 2g: data collapse for height probability. First plot is height probability against h. Second plot is data collapse."""
            

    def avalanche_size_prob(self,N,sizes,W,collapse):
        """Method to do Tasks 3a and 3b
        N = number of grains added once in steady-state
        sizes = list of system sizes to be trialled
        W = moving average window size
        collapse = Boolean: True: perform collapse, False: don't (for efficiency)
        """
        results = []
        for size in sizes:
            self.reset(self.p,size)
            self.stop = True #drive until steady state
            self.drive(10000000)
            self.stop = False
            self.drive(N+1000) #drive for some time longer (+1000 to make sure we're beyond t_c even for small N)
            t_start = self.t_c + 1000 #Point at which you start measuring Probabilities
            s_data = self.avalanche_sizes[t_start:]    
            Num = len(s_data)    
            x,y = logbin(s_data,1.1,False) #True to include events of size 0
            counted = Counter(s_data)
            x_sampled = counted.keys()
            y_sampled = counted.values()
            results.append((x,y,size,x_sampled,y_sampled))
            print "done: ", size
        #MOVING AVERAGE
        results_mov = []
        for x,y1,s,k,l in results:
            collect = []
            N = len(y1)
            for i in range(int(N)):    
                if i < W+1:
                    window = i*2
                elif i == (N-1):
                    window = 0
                elif i > (N-1-W):
                    window = ((N-1)-i)*2
                else:
                    window = (2*W)+1        
                summ = 0
                if i !=0:
                    if i == N-1:
                        summ = y1[i]
                    else:
                        for j in range(i-(window/2),i+(window/2)):
                            summ += y1[j]
                if window != 0:   
                    summ = float(summ)/float(window)
                if summ == 0:
                    summ = y1[0]
                collect.append(summ)
            results_mov.append((x,collect,s))            
        
        plt.figure() #Plot log-binned data to show importance of log-binning
        for j in results:
            if j[2] != 0.0:
                plt.plot(j[0],j[1],label = "Log-binned data")
            for m in range(len(j[3])):
                plt.scatter(j[3][m],float(j[4][m])/float(Num),marker='x',color='black',s=2.0)
        plt.xlabel("s ")
        plt.ylabel("Normalised P(s;L)")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=20)
        plt.show()
        
        plt.figure() #Plot probability on log-log scale for all L
        for j in results_mov:
            plt.plot(j[0],[float(i)/float(Num) for i in j[1]],label = "system size = %s"%j[2])
        plt.xlabel("s ")
        plt.ylabel("(Normalised) P(s;L)")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=20)
        plt.show()
        print """Task 3a: plot avalanche size probabilty for multiple. Figures show effect of log-binning, and the P(s;L) for various L """
                
        if collapse:
            #Very rough overlaps suggest tau = 1.5-1.55, D = 2.0-2.05
            plt.figure()
            for j in results:
                if j[2] != 0.0:
                    tau = 1.5
                    D = 2.1
                    y = []
                    x = []
                    for k in range(len(j[1])):
                        y.append((float(j[1][k])/float(Num))*(j[0][k]**tau))
                        x.append(float(j[0][k])/(float(j[2])**D))
                    plt.plot(x,y,label="system size = %s"%j[2])
            plt.xlabel(r'$s/L^{{D}}, D={0}$'.format(D))
            plt.ylabel(r"$P(s;L)s^{{\tau}}, \tau = {0}$".format(tau))
            plt.xscale("log")
            plt.yscale("log")
            plt.legend(fontsize=20)
            plt.show()
            
            plt.figure()
            for j in results_mov:
                if j[2] != 0.0:
                    tau = 1.5
                    D = 2.1
                    y = []
                    x = []
                    for k in range(len(j[1])):
                        y.append((float(j[1][k])/float(Num))*(j[0][k]**tau))
                        x.append(float(j[0][k])/(float(j[2])**D))
                    plt.plot(x,y,label="system size = %s"%j[2])
            plt.xlabel(r'$s/L^{{D}}, D={0}$'.format(D))
            plt.ylabel(r"$P(s;L)s^{{\tau}}, \tau = {0}$".format(tau))
            plt.xscale("log")
            plt.yscale("log")
            plt.legend(fontsize=20)
            plt.show()
            
            plt.figure()
            for j in results_mov:
                if j[2] != 0.0:
                    tau = 1.54
                    D = 2.15
                    y = []
                    x = []
                    for k in range(len(j[1])):
                        y.append((float(j[1][k])/float(Num))*(j[0][k]**tau))
                        x.append(float(j[0][k])/(float(j[2])**D))
                    plt.plot(x,y,label="system size = %s"%j[2])
            plt.xlabel(r'$s/L^{{D}}, D={0}$'.format(D))
            plt.ylabel(r"$P(s;L)s^{{\tau}}, \tau = {0}$".format(tau))
            plt.xscale("log")
            plt.yscale("log")
            plt.legend(fontsize=20)
            plt.show()
            
            print """Task 3b: Probability Collapse. Figures show collapse for smoothed and unsmoothed data. """
                                    
    def moments(self,sizes,T,collapse):
        """Perform Task 3c.
        sizes = list of system sizes to be trialled
        T = period over which average to be eprformed
        collapse: Boolean value: True: do collapse, False; don't
        """
        moment_results = defaultdict(list)
        for size in sizes:
            self.reset(self.p,size)
            self.stop = True
            self.drive(100000) #drive until steady-state
            #add another T+1 grains to calculate average in recurrent state (+buffer amount)
            self.stop = False
            self.drive(T+(2*self.t_c)+1)
            t0 = int(2*self.t_c)    
            s_data = self.avalanche_sizes[t0:int(t0+T)]    
            for k in [1,2,3,4]:
                s_new = [float(i)**k for i in s_data]
                av_s_k = float(np.sum(s_new))/float(len(s_new))
                moment_results[k].append(av_s_k)        
            print "done for size: ", size
             
        plt.figure() #Plot moments on log-log plot
        for k in moment_results.keys():
            plt.plot(sizes,moment_results[k],label="k ={0}".format(k))
        plt.xlabel("Log(L)")
        plt.ylabel(r'$Log(<s^{k}>)$')
        plt.legend()
        plt.yscale("log")
        plt.xscale("log")
        plt.legend(fontsize=20)
        plt.show()
         
        plt.figure() #Fit straight lines through moments just for visualisation
        gradients = []
        for k in moment_results.keys():
            x = [np.log(i) for i in sizes]
            y =[np.log(i) for i in moment_results[k]]
            fitting = curve_fit(lambda t,a,b: a+(t*b),x,y,p0=[2,0.5])
            a = fitting[0][0]
            b = fitting[0][1]
            gradients.append(b)
            plt.plot([np.log(i) for i in sizes],[(np.log(i)*b)+a for i in sizes],'--',label='Linear Fit: exponent = %s'%b)
            for j in range(len(sizes)):
                plt.scatter(np.log(sizes[j]),np.log(moment_results[k][j]))
        plt.legend(fontsize=20)
        plt.show()
        
        plt.figure() #Find D and tau using moment scaling analysis
        plt.scatter(moment_results.keys(),gradients,marker='x',label = 'data points')
        slope, intercept, r_value, p_value, std_err = stats.linregress(moment_results.keys(),gradients)
        plt.plot(moment_results.keys(),[(slope*i)+intercept for i in moment_results.keys()],'--',label='linear fit')
        print "D = ", slope
        print "tau = ", 1.0-(float(intercept)/float(slope))
        mx = np.mean(moment_results.keys())
        sx2 = np.sum(((moment_results.keys()-mx)**2))
        print "Error in Tau", (std_err*np.sqrt(1./len(moment_results.keys()) + (mx*mx/sx2)))/float(slope)
        print "error in D", std_err*np.sqrt(1.0/sx2)
        plt.xlabel("k")
        plt.ylabel(r'$\gamma_{k}$')
        plt.legend(fontsize=20)
        plt.show(0)
        
        plt.figure() #Plot for obtained guesses of D and tau to see what things look like
        D = 2.215
        tau = 1.54
        for k in moment_results.keys():
            exponent = D*(1+k-tau)
            plt.plot(sizes,[float(i)/(float(sizes[moment_results[k].index(i)])**exponent) for i in moment_results[k]],label = 'moment = %s'%k)
        plt.xlabel("L")
        plt.ylabel(r'$<s^{k}>/(L^{D(1+k-\tau_{s})})$')
        plt.legend(fontsize=20)
        plt.show()

        rsquared = [] #Find corrections to scaling in moments using R^2 linear fiting procedure from previously
        for k in moment_results.keys():
            c0 = np.linspace(0.0001,1,100)
            values = []
            
            plt.figure()
            errors = []
            results = moment_results[k]
            for c in c0:
                if c <= min([float(i)/((float(sizes[results.index(i)]))**(D*(1.0+k-tau))) for i in results]):
                    y = [np.log((float(i)/((float(sizes[results.index(i)]))**(D*(1.0+k-tau))))-c) for i in results]
                    x = [np.log(i) for i in sizes]
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                    values.append((c,r_value,slope))
                    plt.plot(x,y,label="c0=%s"%c)
                    errors.append((c,r_value**2))
            plt.xlabel("Log(L)")
            plt.ylabel(r"$Log(c0 - <s^{k}>/L^{D(1+k-\tau_{s})})$")
            plt.legend(fontsize=20)
            plt.show()
                
            rsquared.append(errors)
            plt.figure()
            plt.plot([i[0] for i in errors], [i[1] for i in errors])
            best_val = errors[[i[1] for i in errors].index(max([i[1] for i in errors]))][0]
            plt.plot([best_val]*100,np.linspace(0,errors[[i[1] for i in errors].index(max([i[1] for i in errors]))][1],100),'--')
            plt.scatter(errors[[i[1] for i in errors].index(max([i[1] for i in errors]))][0],errors[[i[1] for i in errors].index(max([i[1] for i in errors]))][1],marker='x',color='red',label="Best Estimate = %s"%best_val)
            plt.xlabel("c0 estimate")
            plt.ylabel(r"$R^{2} value$")
            plt.legend(fontsize=20)
            plt.show()
            print "For MomentL ", k," The optimum value of c0: ", best_val, " and mu1: ", -1*values[[i[1] for i in errors].index(max([i[1] for i in errors]))][2]
        
        plt.figure()
        for errors in rsquared:
            plt.plot([i[0] for i in errors], [i[1] for i in errors],label='k=%s'%(rsquared.index(errors)+1))
            best_val = errors[[i[1] for i in errors].index(max([i[1] for i in errors]))][0]
            plt.plot([best_val]*100,np.linspace(0.6,errors[[i[1] for i in errors].index(max([i[1] for i in errors]))][1],100),'--')
            plt.scatter(errors[[i[1] for i in errors].index(max([i[1] for i in errors]))][0],errors[[i[1] for i in errors].index(max([i[1] for i in errors]))][1],marker='x',color='red',label=r"$c_{{0}} = {0}$".format(best_val))
        plt.xlabel("c0 estimate")
        plt.ylabel(r"$R^{2} value$")
        plt.ylim([0.6,1.05])
        plt.legend(fontsize=20)
        plt.show()
        print """Task 3c: moment scaling analysis. Figures are moments, moments with straight line fits for visualisation, plot of gradient against k to find D and tau, corrections to scaling test, straight line test for corrections, R^2 test for corrections."""
            
    def reset(self,new_p,new_L):
        """
        Reset all variables in process - allows for user to re-define p and L values if required
        """
        self.__init__(new_p,new_L)  