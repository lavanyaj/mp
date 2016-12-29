import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import time

###################### Global constants ########################
num_instances = 2000
max_iterations = 10000
max_capacity = 100
nflows_per_link = 40
nswitches = 10
np.random.seed(241)
random.seed(241)
################################################################

class MPMaxMin:
    def __init__(self, routes, c):
        
        ########################## Inputs ##############################        
        (self.num_flows, self.num_links) = routes.shape
        self.routes = routes  # incidence matrix for flows / links
        self.c = c            # link capacities
        ################################################################
        # max-min rates
        self.maxmin_level = -1
        self.maxmin_x = self.water_filling()
        # iteration
        self.t = 0                                            
        # sending rates
        self.x = np.zeros((self.num_flows,1))        

        # flows send demand to links
        # links send bottleneck level to flows
        
        # link have rates from previous round,
        # they use rates and flow demands to find new rates
        self.old_rate = self.c
        
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
        rem_links = np.where(rem_cap>0.0, 1, 0)
        level = 0
        wf_links = None
        wf_rates = None
        wf_levels = None
        wf_flows = None
        wf_flow_levels = None
        wf_flow_rates = None
        wf_flow_links = None
        
        bl_rate = 0
        while rem_flows.size != 0:
            link_weights = self.routes.T.dot(weights)
            with np.errstate(divide='ignore', invalid='ignore'):
                bl = np.argmax(np.where(link_weights>0.0, link_weights/rem_cap, -1))
            inc = rem_cap[bl]/link_weights[bl]
            x[rem_flows] = x[rem_flows] + inc*weights[rem_flows]                
            rem_cap = rem_cap - inc*link_weights

            bl_rate += inc
            if (inc > 0):
                # store info for newly bottlenecked links, each time the bottleneck rate jumps
                level += 1
                bl_links = np.where(np.logical_and(rem_cap==0.0, rem_links>0.0))[0]
                #print "bl_links[0]", bl_links[0]
                #print "bl_links[0].shape", bl_links[0].shape
                num_bl = len(bl_links)
                bl_rates = np.full(bl_links.shape, float(bl_rate))
                bl_levels = np.full(bl_links.shape, float(level))
                #print "level ", level, ", bl_links ", bl_links, ", bl_rates ", bl_rates
                #print "rem_cap"
                #print rem_cap.T                
                if wf_links is None and wf_rates is None and wf_levels is None:
                    wf_links = bl_links
                    wf_rates = bl_rates
                    wf_levels = bl_levels
                else:
                    wf_links = np.concatenate((wf_links, bl_links))
                    wf_rates = np.concatenate((wf_rates, bl_rates))
                    wf_levels = np.concatenate((wf_levels, bl_levels))
                    #print "wf_flows ", wf_flows
                    #print "wf_flow_levels ", wf_flow_levels
                    #print "wf_flow_links ", wf_flow_links

            rem_cap = np.where(rem_cap>0.0, rem_cap, 0.0)
            rem_links = np.where(rem_cap>0.0)
            bf = np.nonzero(self.routes[:,bl])[0]

            # store info for newly bottlenecked flows
            new_bf = np.array([f for f in bf if f in rem_flows])
            flow_levels = np.full(new_bf.shape, float(level))
            flow_rates = np.full(new_bf.shape, float(bl_rate))
            flow_links = np.full(new_bf.shape, float(bl))
            if wf_flows is None and wf_flow_levels is None:
                wf_flows = new_bf
                wf_flow_levels = flow_levels
                wf_flow_rates = flow_rates
                wf_flow_links = flow_links
            else:
                wf_flows = np.concatenate((wf_flows, new_bf))
                wf_flow_levels = np.concatenate((wf_flow_levels, flow_levels))
                wf_flow_rates = np.concatenate((wf_flow_rates, flow_rates))
                wf_flow_links = np.concatenate((wf_flow_links, flow_links))
            
            rem_flows = np.array([f for f in rem_flows if f not in bf])
            weights[bf] = 0
        self.maxmin_level = level
        self.wf_links = wf_links
        self.wf_rates = wf_rates
        self.wf_levels = wf_levels
        self.wf_flows = wf_flows
        self.wf_flow_levels = wf_flow_levels
        self.wf_flow_rates = wf_flow_rates
        self.wf_flow_links = wf_flow_links
        
        print("finished waterfilling")
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

        self.last_converged = 0
        for i in range(1, self.maxmin_level):
            level_flows = self.wf_flows[np.where(self.wf_flow_levels==i)]
            level_x = self.x[level_flows]
            level_maxmin = self.maxmin_x[level_flows]
            linf_err = max(np.abs(level_maxmin - level_x))
            if (linf_err > 1e-10):
                # print "level ", i, " hasn't converged yet"
                # print 'x=', level_x.T
                # print 'maxmin=', level_maxmin.T
                # print 'linf_err=', linf_err
                break
            self.last_converged += 1


        self.print_details()
        self.xs = np.column_stack((self.xs, self.x))
        self.errors = np.column_stack((self.errors, max((np.abs(self.x - self.maxmin_x)/self.x)*100)))

    def update_link_to_flow(self):
        new_rate = np.zeros((self.num_links,1))
        #print "\n"
        for l in range(self.num_links):
            #print "\nlink " + str(l)
            flow_indices = np.nonzero(self.routes[:,l])[0]
            temp = list(flow_indices)
            demands = self.flow_to_link[temp,l]
            #print "demands " + str(sorted(demands))
            nflows = len(demands)
            if nflows == 0:
                continue
            #print "old rate " + str(self.old_rate[l])
            sumsat = np.sum(np.where(demands<self.old_rate[l], demands, 0))
            numsat = np.sum(np.where(demands<self.old_rate[l], 1, 0))
            maxsat = np.max(np.where(demands<self.old_rate[l], demands, 0))
            #print "sumsat " + str(sumsat)
            #print "numsat " + str(numsat)
            #print "maxsat " + str(maxsat)
            new_rate[l] = self.c[l] - sumsat + maxsat
            if (numsat < nflows):
                new_rate[l] = (self.c[l]-sumsat)/(nflows - numsat)
                #print "new_rate " + str(new_rate[l])
            #else:
            #print "new_rate " + str(new_rate[l])
                
            bottleneck_level = max(new_rate[l], maxsat)
            #print "bottleneck_level " + str(bottleneck_level)
            
            self.link_to_flow[l, temp] = bottleneck_level
        self.old_rate = new_rate
                    
    def update_flow_to_link(self):
        for f in range(self.num_flows):
            link_indices = np.nonzero(self.routes[f,:])[0]
            self.x[f] = min(self.link_to_flow[link_indices,f])
            temp = list(link_indices)
            self.flow_to_link[f,temp] = self.x[f]
        
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
        print 'last_converged level=', self.last_converged

        rates_hi = []
        rates_lo = []
        maxmin = []
        for i in range(self.last_converged, self.maxmin_level):
            level_flows = self.wf_flows[np.where(self.wf_flow_levels==i)]
            if len(level_flows) == 0:
                continue
            level_x = self.x[level_flows]
            level_maxmin = self.maxmin_x[level_flows]
            if i > self.last_converged:
                rates_lo.append(min(level_x))
                rates_hi.append(max(level_x))
                # if i < self.last_converged+3:
                #     print 'min(x_', i, ')= ', min(level_x)
                #     print 'max(x_', i, ')= ', max(level_x)
            maxmin.append(level_maxmin[0])

        if len(rates_lo) > 0 and len(rates_hi) > 0:
            if self.last_converged > 0:
                if (min(rates_lo) < maxmin[0]):
                    print 'rates of flows in level >', self.last_converged
                    print 'min(x)=', min(rates_lo), ', max(x)=', max(rates_hi), 
                    for diff in range(min(2, len(maxmin))):
                        print ' maxmin(', self.last_converged+diff, ')=', maxmin[diff],
                    assert(min(rates_lo) >= maxmin[0])

        print 'x=', self.x[self.wf_flows].T
        print 'maxmin=', self.maxmin_x[self.wf_flows].T
        print 'l2 error=', np.linalg.norm(self.x - self.maxmin_x)
        print 'linf error=', max(np.abs(self.x - self.maxmin_x))
        print 'x.shape', self.x.shape
        print 'max(x)', max(self.x)
        print 'max(maxmin)', max(self.maxmin_x)
        print 'min(x)', min(self.x)
        print 'min(maxmin)', min(self.maxmin_x)

def gen_random_instance(nswitches, nflows_per_link, safe=True):
    nlinks = nswitches * (nswitches - 1) 
    nflows = nflows_per_link * (nswitches - 1) * 2
    # nflows * path_length = flows_per_link * nlinks

    if (safe):
        A = np.zeros((nflows+nlinks, nlinks))
        for j in range(nlinks):
            flow_index = nflows+j
            A[flow_index, j] = 1
    else:
        A = np.zeros((nflows, nlinks))

    #A = np.zeros((nflows, nlinks))
        
    for i in range(nflows):        
        path_length = random.randint(2, nswitches)
        # lo and hi included
        #print "path length is " + str(path_length)
        #print "nswitches is " + str(nswitches)
        path = np.random.choice(nswitches, path_length, replace=False)
        #print path
        for k in range(path_length-1):
            s1 = path[k]
            s2 = path[k+1]

            j = s1 * (nswitches-1) + s2
            if (s2 > s1):
                # shift down by one cuz we
                # don't have index links from s1 to s1
                j -= 1
            
           # print "link from " + str(s1) + " to " + str(s2) + " is " + str(j)
            A[i, j] = 1

    nflows_by_link = np.sum(A, axis=0)
    print nflows_by_link
    used_links = np.where(nflows_by_link>0.0)[0]
    print used_links
    num_used_links = len(used_links)

    print "original num_links", nlinks
    print "num used links", num_used_links

    # If A is sparse, many links don't have flows
    # make a new matrix B with only used links
    # all used links have at least one flow
    # we also want to add a one link only flow
    # for each link
    # if (safe):
    #     B = np.zeros((nflows+num_used_links, num_used_links))
    #     for j in range(num_used_links):
    #         flow_index = nflows+j
    #         B[flow_index, j] = 1
    #         old_index = used_links[j]
    #         for i in range(nflows):
    #             B[i, j] = A[i, old_index]
    # else:
    #     B = np.zeros((nflows, num_used_links))
    #     for j in range(num_used_links):
    #         old_index = used_links[j]
    #         B[:, j] = A[:, old_index]


    # print "shape of original matrix ", A.shape()
    # print "shape of new matrix ", B.shape()
    # A = B
    
    c = np.ones((num_used_links, 1))
    return A,c

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

# def main1():
#     np.random.seed(241)
#     plt.close("all")
    
#     print("5 steps of max min for random_instance(5)")
#     A,c = gen_random_instance(nswitches=3, nflows_per_link=1)
#     print A
#     print c
#     wf_maxmin = MPMaxMin(A, c)
#     print wf_maxmin.maxmin_x
#     print wf_maxmin.maxmin_level
#     for i in range(5):
#         wf_maxmin.step()    
    #error5 = wf_maxmin.errors
    
def main():
    plt.close("all")
    f = open("results.txt", "w")
    #A = np.array([[1,0],
    #              [1,1],
    #              [1,0],
    #              [0,1]])       
    #c = np.array([[1.0, 1.0]]).T
    #num_steps = 10000
    #num_instances = 10

    print max_iterations, " steps of max min for ", num_instances,\
        " random instance with ", nflows_per_link,\
        " flows/ link, ", nswitches, " switches"
    for i in range(num_instances):
        print 'instance ', i
        A,c = gen_random_instance(nswitches=nswitches,\
                                  nflows_per_link=nflows_per_link, safe=True)  
        wf_maxmin = MPMaxMin(A, c)
        
        steps_to_converge = -1
        start_time = time.time()
        for j in range(1, max_iterations+1):
            wf_maxmin.step()
            linf_err = max(np.abs(wf_maxmin.x - wf_maxmin.maxmin_x))
            l2_err = np.linalg.norm(wf_maxmin.x - wf_maxmin.maxmin_x)
            print "instance ", i, "step ", j, ", linf_err ",\
                linf_err, ", l2_err ", l2_err,\
                ", elapsed time", (time.time()-start_time)
            sys.stdout.flush()
            if linf_err < 1e-10:
                steps_to_converge = j
                break
                
        if (steps_to_converge > 0):
            f.write("instance %d converged after %d steps.\n"%(i, steps_to_converge))
            print "instance ", i, "converged after ", steps_to_converge, " steps."
        else:
            f.write("instance %d converged after %d steps.\n"%(i, max_iterations))
            print "instance ", i, "did not converge after ", max_iterations, " steps."
        wf_maxmin.print_details()
    f.close()   
    
                    
if __name__ == '__main__':
    main()

