#
import numpy as np
import itertools
import sys
import os as os
import cPickle as pickle
import math 




class SignalNet:

	def __init__(self, name, prob_writer, prob_eraser, num_writer, num_eraser, num_flavor):

		self.f0 = -2 # epsilon
		self.W  = 10
		self.probthreshold = 0.5
		self.prob_writer = prob_writer
		self.prob_eraser = prob_eraser
		self.num_writer = num_writer
		self.num_eraser = num_eraser
		self.num_flavor = num_flavor
		self.Ematrix = self.connectivity
		self.flavorset = np.array([1.0, -6.0]) # mu. note if f>> f0 (mu >> epsilon), then Prob -> 1; if f<<f0 then Prob-> 0
		#self.flavorset = np.linspace(self.f0, 0.0, num = num_flavor ) # chemical potential
		self.MC_prob = self.sample_prob

		

		

	
	def connectivity(self):  

		
		E = np.zeros((self.num_writer + self.num_eraser, self.num_writer + self.num_eraser), dtype = float) + self.W
		
		arr_writer = np.arange(self.num_writer)
		arr_eraser = np.arange(self.num_eraser) + self.num_writer

		np.random.shuffle(arr_writer)
		np.random.shuffle(arr_eraser)
		#print arr_writer

		for i in range(self.num_writer):

			
			mask_writer = np.where((np.random.random_sample((self.num_writer - i - 1,)) - self.prob_writer) <= 0)[0] + i + 1
			#print mask_writer
			flavors_writer_writer =  np.random.choice(self.flavorset, len(mask_writer))
			#print flavors_writer_writer

			E[arr_writer[i], mask_writer] = self.f0 - flavors_writer_writer

		for i in range(self.num_eraser):

			mask_eraser = np.where((np.random.random_sample((self.num_writer,)) - self.prob_eraser) <= 0)[0]
			flavors_eraser_writer =  np.random.choice(self.flavorset, len(mask_eraser))

			E[arr_eraser[i], mask_eraser] = self.f0 - flavors_eraser_writer



		return E

	

	def config_prob(self, config):

		'''
		For a given Monte Carlo sample (i.e. energy matrix and the underlying network), formulate the joint pdf. 
		Note that this depends on x, which is stored as a binary list. Also assumes that nodes with not incident edges 
		are assumed to have activation prob 1
		'''

		Emodel = self.Ematrix()
		#end_node_counter = 0
		for i in range(self.num_writer + self.num_eraser):

			arr = np.where(Emodel[i,:] != self.W)[0]
			#end_node_counter += np.linspace(self.W, self.W, num = self.num_writer + self.num_eraser)

			tmp_writer = 0.0
			tmp_eraser = 0.0
			prob_writer = 0.0
			prob_eraser = 0.0
			for j in range(len(arr)):
				if arr[j] < self.num_writer:
					tmp_writer += config[arr[j]] * np.exp(-Emodel[i, arr[j]])
				else:
					tmp_eraser += config[arr[j]] * np.exp(-Emodel[i, arr[j]])
			prob_writer = tmp_writer / (1 + tmp_writer)
			prob_eraser = 1 / (1 + tmp_eraser)

		
		#return prob_writer * prob_eraser * 0.5**end_node_counter
		return prob_writer * prob_eraser







	
	def sample_prob(self):

		
		'''
		This function calculates the joint pdf of all 2^n configurations in a given Monte Carlo sample
		'''
		# first generate all configurations
		lst = map(list, itertools.product([0, 1], repeat = self.num_writer+ self.num_eraser))
		config_prob_all = np.zeros((2**(self.num_writer+ self.num_eraser),2))



		for i in range(len(lst)):
			config_prob_all[i,0] = self.config_prob(lst[i])
			if config_prob_all[i,0] <= self.probthreshold:
				config_prob_all[i,1] = 1  # 1 for config. w/ prob <=0.5, discard


		
		return config_prob_all
			




		




'''
Begin main program: need to specify nproteins, alpha (i.e. fraction of writer), and beta (\equiv #flavors/ nproteins)
'''



nproteins = int(sys.argv[1])  # argument i in the batch file, number of proteins
alpha = float(sys.argv[2]) # argument j in the batch file, fraction of writers
#beta = float(sys.argv[3]) # argument k in the batch file,  # flavors = nproteins * beta


 

# Directory 
dir_name='./'


cutoff = 0.5  # config. counts only when its prob is greater than or equal to cutoff
pw = 0.5
pe = 0.5
nw = int(math.floor(nproteins * alpha))
ne = nproteins - nw
#nf = int(math.floor(nproteins * beta))
nf = 10 
MCsamples = 500


#file_name = 'Capacity_n' + str(nproteins) + '_alpha' + str(alpha) + '_beta' + str(beta) + '_MCsteps' + str(MCsamples)
file_name = 'MC_Capacity_n' + str(nproteins) + '_alpha' + str(alpha) + '_nf' + str(nf) + '_MCsteps' + str(MCsamples) + '_pw' + str(pw) +  '_pe' + str(pe)
parameters = ['nproteins:', nproteins, 'nwriters:', nw, 'nerasers:', ne, 'MCsteps:', MCsamples, 'nflavors:', nf, 'prob_w:', pw, 'prob_e:', pe]



tn = SignalNet(file_name, pw, pe, nw, ne, nf)

#print tn.Ematrix()


# loop over all MC samples

p_configurations = np.zeros((2**nproteins,2), dtype = float)
p_configurations_all = np.zeros(2**nproteins, dtype = float)



if nproteins < 10:
	noupdate_threshold = 1000
else:
	noupdate_threshold = 10000


capacityinit = 4.0
capacity_MC = 0  # capacity after Monte Carlo average
capacity_all = np.array([])


for i in range(MCsamples):
	noupdate = 0 # no-update counter
	capacity = capacityinit
	p_configurations = tn.MC_prob()
	
	while noupdate <= noupdate_threshold:
		tmp = np.random.choice(2**nproteins, int(capacity), replace = False) # generate a set of config of size capacity

		if np.all(p_configurations[tmp, 1] == np.linspace(0,0,num= capacity)): # keep this set, record, and move on
			capacity += 1 # move on
		else:
			noupdate += 1

	#print capacity
	capacity_MC += capacity
	capacity_all = np.append(capacity_all, capacity)
	p_configurations_all += p_configurations[:,0]



capacity_MC = capacity_MC / MCsamples
p_configurations_all = p_configurations_all / MCsamples





# file I/O
name = os.path.join(dir_name, file_name)
pickle.dump( {'ConfigurationProb':p_configurations_all, 'Parameters':parameters, 'MCavgCapacity': capacity_MC, 'MCCapacityall': capacity_all}, open(name, 'wb'))


#print parameters
#print capacity_MC


