import sys
import random
import matplotlib.pyplot as plt
import numpy as np

###################### Global constants ########################
num_instances = 1
max_iterations = 10000
max_capacity = 100
################################################################

class MPMaxMin:
    def __init__(self, routes, c):
        
        ########################## Inputs ##############################        
        (self.num_flows, self.num_links) = routes.shape
        self.routes = routes  # incidence matrix for flows / links
        self.c = c            # link capacities
        ################################################################
        # max-min rates
        self.maxmin_x = self.water_filling()
        # iteration
        self.t = 0                                            
        # sending rates
        self.x = np.zeros((self.num_flows,1))        
         
        # flow to link messages
        self.flow_to_link = self.routes * max_capacity * 1.0
        # link to flow messages
        self.link_to_flow = np.zeros((self.num_links, self.num_flows))
           
        # timeseries
        self.xs = self.x
        self.errors = max(np.abs(self.x - self.maxmin_x))
                      
    def water_filling(self): 
        weights = np.ones((self.num_flows, 1))
        x = np.zeros((self.num_flows,1))
        rem_flows = np.array(range(self.num_flows))
        rem_cap = np.array(self.c, copy=True)   
        while rem_flows.size != 0:
            link_weights = self.routes.T.dot(weights)
            with np.errstate(divide='ignore', invalid='ignore'):
                bl = np.argmax(np.where(link_weights>0.0, link_weights/rem_cap, -1))
            inc = rem_cap[bl]/link_weights[bl]
            x[rem_flows] = x[rem_flows] + inc*weights[rem_flows]                
            rem_cap = rem_cap - inc*link_weights
            rem_cap = np.where(rem_cap>0.0, rem_cap, 0.0)       
            bf = np.nonzero(self.routes[:,bl])[0]
            rem_flows = np.array([f for f in rem_flows if f not in bf])
            weights[bf] = 0 
        return x
                                                                           
    def step(self):
        self.t += 1
        
        #print '$$$$$$$$ t = ', self.t
        #print 'flow_to_link='
        #print self.flow_to_link
        self.update_link_to_flow()
        #print 'link_to_flow='
        #print self.link_to_flow
        self.update_flow_to_link()
              
        self.xs = np.column_stack((self.xs, self.x))
        self.errors = np.column_stack((self.errors, max(np.abs(self.x - self.maxmin_x))))

    def update_link_to_flow(self):
        for l in range(self.num_links):
            flow_indices = np.nonzero(self.routes[:,l])[0]
            for ind in range(len(flow_indices)):
                temp = list(flow_indices)
                del temp[ind]
                f = flow_indices[ind]
                if len(temp) == 0:
                    self.link_to_flow[l,f] = self.c[l]
                else:  
                    demands = self.flow_to_link[temp,l]
                    self.link_to_flow[l,f] = self.fair_share(demands, self.c[l])

                    
    def update_flow_to_link(self):
        for f in range(self.num_flows):
            link_indices = np.nonzero(self.routes[f,:])[0]
            self.x[f] = min(self.link_to_flow[link_indices,f])
            for ind in range(len(link_indices)):
                temp = list(link_indices)
                del temp[ind]
                l = link_indices[ind]
                if len(temp) == 0:
                    self.flow_to_link[f,l] = max_capacity
                else:
                    fair_shares = self.link_to_flow[temp,f]
                    self.flow_to_link[f,l] = min(fair_shares)

    def fair_share(self, demands, cap):
        demands = sorted(demands)
        demands.append(max_capacity)
        nflows = len(demands)
        level = 0.0
        mycap = float(cap)
        for f in range(nflows):
            rem_share = mycap / (nflows - f)
            inc = np.min([rem_share, demands[f]])
            level += inc
            mycap -= inc*(nflows-f)
            demands = demands - inc
        return level
                                        
    def print_details(self):
        print 'iteration=', self.t
        print 'x=', self.x
        print 'maxmin=', self.maxmin_x
        print 'l2 error=', np.linalg.norm(self.x - self.maxmin_x)
        print 'linf error=', max(np.abs(self.x - self.maxmin_x))

def gen_random_bipartite_instance(nports, nflows):
    A = np.zeros((nflows, 2*nports))
    for i in range(nflows):
        src = np.random.randint(nports)
        dst = np.random.randint(nports)
        A[i, src] = 1
        A[i, nports+dst] = 1
    c = np.ones((2*nports, 1))
    return A,c

def gen_large_chain(nports):
    nflows = nports*(nports+1)/2
    A = np.zeros((nflows, 2*nports))
    f = 0
    for j in range(nports):
        for i in range(j,nports):
            A[f,i] = 1
            A[f,nports+j] = 1
            f += 1
    c = np.ones((2*nports, 1))
    return A,c

def main():
    np.random.seed(241)
    plt.close("all")
    
    #A = np.array([[1,0],
    #              [1,1],
    #              [1,0],
    #              [0,1]])       
    #c = np.array([[1.0, 1.0]]).T
    
    #A, c = gen_random_bipartite_instance(50, 130)
    #wf_maxmin = MPMaxMin(A,c)
    #for i in range(10):
    #    wf_maxmin.step()
        
    A,c = gen_large_chain(5)  
    wf_maxmin = MPMaxMin(A, c)
    for i in range(20):
        wf_maxmin.step()    
    error5 = wf_maxmin.errors

    A,c = gen_large_chain(10)  
    wf_maxmin = MPMaxMin(A, c)
    for i in range(20):
        wf_maxmin.step()   
    error10 = wf_maxmin.errors 
                                   
    A,c = gen_large_chain(50)  
    wf_maxmin = MPMaxMin(A, c)
    for i in range(20):
        wf_maxmin.step()   
    error50 = wf_maxmin.errors 

    A,c = gen_large_chain(100)  
    wf_maxmin = MPMaxMin(A, c)
    for i in range(20):
        wf_maxmin.step()   
    error100 = wf_maxmin.errors         
                                                                                                
    plt.figure('link rate')
    plt.title('link rates')
    plt.plot(wf_maxmin.xs.T.dot(wf_maxmin.routes))
    plt.legend(range(wf_maxmin.num_links))
    plt.show()
                    
    plt.figure('flow rate')
    plt.title('flow rates')
    plt.plot(wf_maxmin.xs.T)
    plt.legend(range(wf_maxmin.num_flows))
    plt.show()
    
    plt.figure('error')
    plt.title('error')
    plt.semilogy(wf_maxmin.errors.T)
    plt.ylabel('max element-wise error')
    plt.xlabel('iteration')
    plt.show()
 
    plt.figure('error2')
    plt.title('error')
    plt.semilogy(error5.T)
    plt.semilogy(error10.T)
    plt.semilogy(error50.T)
    plt.semilogy(error100.T)
    plt.legend(['N=5', 'N=10', 'N=50', 'N=100'])
    plt.ylabel('max element-wise error')
    plt.xlabel('iteration')
    plt.show()
                   
    sys.stdout.flush()
                    
if __name__ == '__main__':
    main()

