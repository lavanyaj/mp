import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import mpmath

###################### Global constants ########################
num_instances = 200000
max_iterations = 200
max_capacity = 100
nflows_per_link = 100
nswitches = 20
np.random.seed(241)
random.seed(241)
mpmath.dps = 100
zero53 = mpmath.mpmathify(0) #mpf(str("".join(['0']*100)))
################################################################

class MPMaxMin:
    def __init__(self, routes, c):
        
        ########################## Inputs ##############################        
        (self.num_flows, self.num_links) = routes.rows, routes.cols
        self.routes = routes  # incidence matrix for flows / links
        self.c = c            # link capacities
        ################################################################
        # max-min rates
        self.maxmin_level = -1
        self.maxmin_x = self.water_filling()
        # iteration
        self.t = 0                                            
        # sending rates
        self.x = mpmath.zeros(self.num_flows,1)
        self.old_demands = mpmath.zeros(self.num_flows,1)
        
        # flows send demand to links
        # links send bottleneck level to flows
        
        # link have rates from previous round,
        # they use rates and flow demands to find new rates
        self.old_rate = self.c
        self.old_num_unsat = mpmath.zeros(self.num_links,1)
        self.old_old_rate = self.c
        
        # flow to link messages
        self.flow_to_link = self.routes * max_capacity * 1.0
        # link to flow messages
        self.link_to_flow = mpmath.zeros(self.num_links, self.num_flows)
           
        # timeseries
        self.xs = self.x
        #self.errors = max(mpmath.norm(self.x - self.maxmin_x, p=1))
                      
    def water_filling(self): 
        weights = mpmath.ones(self.num_flows, 1)
        x = mpmath.zeros(self.num_flows,1)
        rem_flows = np.array(range(self.num_flows))
        rem_cap = self.c.copy() #np.array(self.c, copy=True)
        level = 0
        while rem_flows.size != 0:
            level += 1

            # number of rem flows on all links
            link_weights = self.routes.T * weights
            print "link_weights nrows", link_weights.rows, " ncols", link_weights.cols
            print "rem_cap nrows", rem_cap.rows, " ncols", rem_cap.cols
            print "Weight of link 4 is ", link_weights[4], " (127?)"
            
            #don't want to select links where weight or rem_cap is 0
            #adj_link_weights = np.where(link_weights>zero53, link_weights, mpmath.mpf('-1')) # avoid divide by zero
            #adj_rem_cap = np.where(rem_cap>zero53, rem_cap, mpmath.mpf('-1'))
            #assert((adj_rem_cap<=self.c).all())            
            #bl = np.argmax(np.where(adj_link_weights>zero53, adj_link_weights/adj_rem_cap, mpmath.mpf('-1')))
            #assert(link_weights[bl] > zero53) 
            #fs = list(np.where(adj_link_weights>zero53, adj_link_weights/adj_rem_cap, mpmath.mpf('-1')))
            #print "level ", level, " bottleneck link is ", bl, ", rem_cap is ", rem_cap[bl],
            #print " link_weight is ", link_weights[bl]
            fair_shares = mpmath.zeros(self.num_links, 1)
            max_inv = 0
            inc = 0
            bl = -1
            for l in range(self.num_links):
                #print "Link ", l
                if (link_weights[l,0] > zero53 and rem_cap[l,0] > zero53):
                    fair_shares[l,0] = rem_cap[l,0]/link_weights[l,0]
                    inv = link_weights[l,0]/rem_cap[l,0]
                    if (inv > max_inv):
                        inc = fair_shares[l,0]
                        bl = l
                        max_inv = inv

            print "bottleneck link is ", bl, " with inc ", inc
            assert(bl != -1)

            for f in rem_flows:
                x[f,0] = x[f,0] + inc*weights[f,0]

            print "updated flow rates"
            
            neg_cap = []
            for l in range(self.num_links):
                rem_cap[l,0] = rem_cap[l,0] - inc*link_weights[l,0]
                if (rem_cap[l,0] < zero53):
                    neg_cap.append((rem_cap[l,0], type(rem_cap[l,0])))
                    rem_cap[l,0] = zero53
            print "updated link capacities"
                    
            if (len(neg_cap) > 0):
                print "links with neg. rem_cap ", neg_cap
                #assert((rem_cap >= zero53).all())

            #rem_cap = np.where(rem_cap>zero53, rem_cap, zero53)       
            bf = []
            for f in range(self.num_flows):
                if self.routes[f,bl] > zero53:
                    bf.append(f)
            print "got list of ", len(bf), " flows on bottleneck link"
            
            #bf = np.nonzero(self.routes[:,bl])[0]
            rem_bf = np.array([f for f in rem_flows if f in bf])
            rem_flows = np.array([f for f in rem_flows if f not in bf])
            print "level ", level, " bottleneck link is ", bl,
            print ", inc was ", inc,
            print ", ", len(rem_bf), " rem_bf flows: ",
            if len(rem_bf) > 0:
                print " bottleneck flow has rate ", x[rem_bf[0]]
            else:
                print "\n"

            for f in bf:
                weights[f, 0] = zero53

        self.maxmin_level = level
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
        self.print_details()
              
        self.xs = np.column_stack((self.xs, self.x))
        #self.errors = np.column_stack(self.errors, max((np.abs(self.x - self.maxmin_x)))


    def update_link_to_flow(self):
        new_rate = np.zeros((self.num_links,1), dtype=mpmath.mpf)
        #new_rate = np.zeros((self.num_links,1))
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
            assert(numsat < nflows)
            if (numsat < nflows):
                new_rate[l] = (self.c[l]-sumsat)/(nflows - numsat)
                #print "new_rate " + str(new_rate[l])
            #else:
            #print "new_rate " + str(new_rate[l])
                
            bottleneck_level = max(new_rate[l], maxsat)
            #print "bottleneck_level " + str(bottleneck_level)
            
            self.link_to_flow[l, temp] = bottleneck_level
        self.old_rate = new_rate

        
#     def update_link_to_flow(self):
#         new_rate = np.zeros((self.num_links,1), dtype=mpmath.mpf)
#         new_num_unsat = np.zeros((self.num_links,1))

        
#         #print "\n"
#         for l in range(self.num_links):
#             flow_indices = np.nonzero(self.routes[:,l])[0]
#             temp = list(flow_indices)
#             demands = self.flow_to_link[temp,l]
#             nflows = len(demands)
#             if nflows == 0:
#                 continue
#             sumsat = 0
#             numsat = 0
#             maxsat = 0
#             for ind in range(len(flow_indices)):
#                 f = flow_indices[ind]
#                 dem = self.flow_to_link[f, l]
#                 adj_rate = self.old_rate[l]
#                 # if flow was labeled SAT in last round
#                 # adjust fair share rate to assume it was unsat

# #                 print "link ", l, " -> flow", f
# #                 print "flow demand d(t) ", dem
# #                 print "old link rate r(t-1) ", self.old_rate[l]
# #                 print "old flow demand d(t-1) ", self.old_demands[f]
# #                 print "old old link rate r(t-2) ", self.old_old_rate[l]

#                 if self.old_old_rate[l] < self.c[l]:
#                     if self.old_demands[f]<self.old_old_rate[l]:
#                         old_cap = self.old_rate[l]*self.old_num_unsat[l]
#                         adj_cap = old_cap + self.old_demands[f]
#                         adj_rate = adj_cap/(self.old_num_unsat[l]+1)
#                         #print "flow was sat in last round, adjust rate to ", adj_rate

#                 if dem < adj_rate:
#                     #print "flow's demand ", dem, " is less than adjusted rate ", adj_rate
#                     sumsat += dem
#                     numsat += 1
#                     maxsat = max(maxsat, dem)            
#             new_rate[l] = self.c[l] - sumsat + maxsat
#             assert(numsat < nflows)
#             if (numsat < nflows):
#                 new_rate[l] = (self.c[l]-sumsat)/(nflows - numsat)
#                 #print "new_rate " + str(new_rate[l])
#             #else:
#             #print "new_rate " + str(new_rate[l])
#             #print "sumsat ", sumsat, ", numsat ", numsat, ", maxsat ", maxsat
#             #print "new rate ", new_rate[l]

#             bottleneck_level = max(new_rate[l], maxsat)
#             #print "bottleneck_level " + str(bottleneck_level)
#             new_num_unsat[l] = nflows - numsat
#             self.link_to_flow[l, temp] = bottleneck_level
#         self.old_old_rate = self.old_rate
#         self.old_rate = new_rate
#         self.old_num_unsat = new_num_unsat
                    
    def update_flow_to_link(self):
        for f in range(self.num_flows):
            link_indices = np.nonzero(self.routes[f,:])[0]
            self.x[f] = min(self.link_to_flow[link_indices,f])
            temp = list(link_indices)
            self.old_demands[f] = self.flow_to_link[f,temp[0]]
            self.flow_to_link[f,temp] = self.x[f]
            new_demands = self.x[f]

        
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
        print 'x=', self.x.T
        print 'maxmin=', self.maxmin_x.T
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
        A = mpmath.zeros(nflows+nlinks, nlinks)
        for j in range(nlinks):
            flow_index = nflows+j
            A[flow_index, j] += 1.0
    else:
        A = mpmath.zeros(nflows, nlinks)

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
            A[i, j] += 1.0

    print "nflows=", nflows
    print "original num_links", nlinks
    c = mpmath.ones(nlinks, 1)
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

def main1():
    np.random.seed(241)
    plt.close("all")
    
    print("10 steps of max min for random_instance(5)")
    A,c = gen_random_instance(nswitches=3, nflows_per_link=1)
    print A
    print c
    wf_maxmin = MPMaxMin(A, c)
    print wf_maxmin.maxmin_x
    print wf_maxmin.maxmin_level
    for i in range(10):
        wf_maxmin.step()    
    #error5 = wf_maxmin.errors
    
def main():
    np.random.seed(241)

    f = open("results.txt", "w")
    #A = np.array([[1,0],
    #              [1,1],
    #              [1,0],
    #              [0,1]])       
    #c = np.array([[1.0, 1.0]]).T
    print max_iterations, " steps of max min for ", num_instances,\
        " random instance with ", nflows_per_link,\
        " flows/ link, ", nswitches, " switches"
    for i in range(num_instances):
        
        A,c = gen_random_instance(nswitches=nswitches,\
                                  nflows_per_link=nflows_per_link, safe=True)  
        wf_maxmin = MPMaxMin(A, c)
        steps_to_converge = -1
        start_time = time.time()
        for j in range(1, max_iterations+1):
            wf_maxmin.step()
            linf_err = max(np.abs(wf_maxmin.x - wf_maxmin.maxmin_x))
            l2_err = np.linalg.norm(wf_maxmin.x - wf_maxmin.maxmin_x)
            if j%10 == 0 or linf_err < 1e-8:
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
            f.write("instance %d did not converge after %d steps.\n"%(i, max_iterations))
            print "instance ", i, "did not converge after ", max_iterations, " steps."
        wf_maxmin.print_details()
    f.close()   
                    
if __name__ == '__main__':
    main()

