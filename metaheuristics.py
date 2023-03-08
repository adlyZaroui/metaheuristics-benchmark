import warnings
warnings.filterwarnings('ignore')

# source : https://github.com/Naereen/notebooks/blob/master/Simulated_annealing_in_Python.ipynb
class simulated_annealing:
    name = 'Simulated Annelaing'
    
    def Metropolis_acceptance(x,y,T):
        import numpy as np
        return int(y < x) + np.exp(- (y - x) / T)*int(y >= x)
    
    recommended_temperature = lambda x: max(0.01, min(1, 1 - x))
    
    def recommended_random_neighbour(x, interval, fraction=1):
            """Move a little bit x, from the left or the right."""
            import numpy as np
            rng = np.random.default_rng()
            d = x.shape[0]
            amplitude = np.array([(max((a,b)) - min((a,b))) * fraction / 10 for a,b in interval])
            delta = (-amplitude/2.) + amplitude * rng.random(size = d)
            clip = lambda x: [max(min(x, b), a) for (a,b),x in zip(interval,x)]
            return np.array(clip(x + delta))
    
    def __init__(self, cost_function,
                 p=Metropolis_acceptance,
                 temperature=recommended_temperature,
                 random_neighbour=recommended_random_neighbour,
                 maxsteps = 100):
        
        import numpy as np
        
        self.cost_function = cost_function
        self.d = cost_function.d
        self.input_domain = cost_function.input_domain
        self.p = p
        self.temperature = temperature
        self.random_neighbour = random_neighbour
        self.maxsteps = maxsteps
        
        self.state = None
        self.costs = None
    
    def update_state(self, state): self.state = state
    def update_costs(self, costs): self.costs = costs
    
    def optimize(self,
                 debug=False,
                 show_progress=True):
        
        import numpy as np

        rng = np.random.default_rng()
        state = np.array([a + (b - a) * rng.random() for a,b in self.input_domain])
        cost = self.cost_function(state)
        states, costs = [], []
    
        if show_progress:
            from tqdm import tqdm
            for step in tqdm(range(self.maxsteps)):
        
                fraction = step / float(self.maxsteps)
        
                T = self.temperature(1-fraction)
        
                new_state = self.random_neighbour(state, self.input_domain, fraction)
                new_cost = self.cost_function(new_state)
        
                if debug: print("Step = " + str(step) + ", state = " + str(state) + ", cost = "
                        + str(cost) + ", new_state = " + str(new_state) + "\n")
                        
                if self.p(self.cost_function(state),self.cost_function(new_state),T) > rng.random():
                    state, cost = new_state, new_cost
                states.append(state)
                costs.append(cost)
        
                self.update_state(state)
                self.update_costs(costs)
            
            return state, self.cost_function(state), states, costs 
        
        for step in range(self.maxsteps):
            fraction = step / float(self.maxsteps)
            T = self.temperature(1-fraction)
            new_state = self.random_neighbour(state, self.input_domain, fraction)
            new_cost = self.cost_function(new_state)
            if debug: print("Step = " + str(step) + ", state = " + str(state) + ", cost = "
                            + str(cost) + ", new_state = " + str(new_state) + "\n")
            if self.p(self.cost_function(state),self.cost_function(new_state),T) > rng.random():
                state, cost = new_state, new_cost
            states.append(state)
            costs.append(cost)
            self.update_state(state)
            self.update_costs(costs)
        return state, self.cost_function(state), states, costs
    
    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(range(self.maxsteps),self.costs)
        plt.xlabel('iterations')
        plt.ylabel('cost')
        plt.show()
        
    def evaluate(self,
                 maxiter=30,
                 show_progress=True):
        
        import numpy as np
        from time import time
        costs = np.zeros(maxiter)
        times = np.zeros(maxiter)
        if show_progress:
            from tqdm import tqdm
            for i in tqdm(range(maxiter)):
                t1 = time()
                self.optimize(debug=False, show_progress=False)
                t2 = time()
                costs[i] = self.cost_function(self.state)
                times[i] = t2 - t1
            return (np.min(costs),np.max(costs),np.std(costs),np.mean(times))
        for i in range(maxiter):
            t1 = time()
            self.optimize(debug=False, show_progress=False)
            t2 = time()
            costs[i] = self.cost_function(self.state)
            times[i] = t2 - t1
        return (np.min(costs),np.max(costs),np.std(costs),np.mean(times))

        
# --------------------------------------------------------------------------------------------------------------------+
        
# source : https://github.com/nathanrooy/differential-evolution-optimization
class differential_evolution:

    name = 'Differential Evolution'
    
    def __init__(self, 
                 cost_function,
                 mutate=0.5,
                 recombination=0.5,
                 popsize=20,
                 maxsteps=100):
    
        import numpy as np
        
        self.cost_function = cost_function
        self.d = cost_function.d
        self.input_domain = cost_function.input_domain
        self.popsize = popsize
        self.mutate = mutate
        self.recombination = recombination
        self.maxsteps = maxsteps
        
        self.state = None
        self.costs = None
        
    def update_state(self, state): self.state = state
    def update_costs(self, costs): self.costs = costs
    
    def optimize(self,
                 debug=False,
                 show_progress=True):
    
    
        import numpy as np
        clip = lambda x: [max(min(x, b), a) for (a,b),x in zip(self.input_domain,x)]

        rng = np.random.default_rng()
        population = np.array([[rng.uniform(a,b) for a,b in self.input_domain] for i in range(self.popsize)])
        gen_avg_scores = []
        states, costs = [], []
    
        if show_progress:
            from tqdm import tqdm
            for i in tqdm(range(self.maxsteps)):
                gen_scores = []
                
                # cycle through each individual in the population
                for j in range(self.popsize):
        
                    #--- MUTATION ---------------------+
                    # select three random vector index positions [0, popsize), not including current vector j
                    candidates = list(range(self.popsize))
                    candidates.remove(j)
                    random_index = rng.choice(candidates, 3, replace=False)

                    x_1, x_2, x_3,  = population[random_index[0]], population[random_index[1]], population[random_index[2]]
                    x_t = population[j]


                    x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

                    v_donor = [x_1_i + self.mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
                    v_donor = clip(v_donor)

                     #--- RECOMBINATION ----------------+
                    v_trial = []
                    for k in range(self.d):
                        crossover = rng.uniform()
                        if crossover <= self.recombination:
                            v_trial.append(v_donor[k])
                        else:
                            v_trial.append(x_t[k])
                    
                    #--- GREEDY SELECTION -------------+

                    score_trial  = self.cost_function(np.array(v_trial))
                    score_target = self.cost_function(np.array(x_t))

                    if score_trial < score_target:
                        population[j] = v_trial
                        gen_scores.append(score_trial)
                    else:
                        gen_scores.append(score_target)


                costs.append(min(gen_scores))
                gen_sol = population[gen_scores.index(min(gen_scores))]
                state = gen_sol
                
                states.append(state)
                
                if debug:
                    print ('generation : ' + str(i) + 
                       ', generation average cost : ' + str(gen_avg) + 
                       ', generation best cost : ' + str(gen_best), +
                       ', reached at candidate : ' + str(gen_sol) + '\n')
                    
            
            self.update_state(state)
            self.update_costs(costs)
            
            return state, self.cost_function(state), costs
        
        for i in range(self.maxsteps):
            gen_scores = []   
            for j in range(self.popsize):
                candidates = list(range(self.popsize))
                candidates.remove(j)
                random_index = rng.choice(candidates, 3, replace=False)
                x_1, x_2, x_3,  = population[random_index[0]], population[random_index[1]], population[random_index[2]]
                x_t = population[j]
                x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]
                v_donor = [x_1_i + self.mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
                v_donor = clip(v_donor)
                v_trial = []
                for k in range(self.d):
                    crossover = rng.uniform()
                    if crossover <= self.recombination:
                        v_trial.append(v_donor[k])
                    else:
                        v_trial.append(x_t[k])
                score_trial  = self.cost_function(np.array(v_trial))
                score_target = self.cost_function(np.array(x_t))
                if score_trial < score_target:
                    population[j] = v_trial
                    gen_scores.append(score_trial)
                else:
                    gen_scores.append(score_target)
            costs.append(min(gen_scores))
            gen_sol = population[gen_scores.index(min(gen_scores))]
            state = gen_sol
            states.append(state)
            if debug:
                print ('generation : ' + str(i) + 
                       ', generation average cost : ' + str(gen_avg) + 
                       ', generation best cost : ' + str(gen_best), +
                       ', reached at candidate : ' + str(gen_sol) + '\n')
        self.update_state(state)
        self.update_costs(costs)    
        return state, self.cost_function(state), costs
        
        
    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(range(self.maxsteps),self.costs)
        plt.xlabel('iterations')
        plt.ylabel('cost')
        plt.show()
        
    def evaluate(self,
                 maxiter=30,
                 show_progress=True):
        
        import numpy as np
        from time import time
        costs = np.zeros(maxiter)
        times = np.zeros(maxiter)
        if show_progress:
            from tqdm import tqdm
            for i in tqdm(range(maxiter)):
                t1 = time()
                self.optimize(debug=False, show_progress=False)
                t2 = time()
                costs[i] = self.cost_function(self.state)
                times[i] = t2 - t1
            return (np.min(costs),np.max(costs),np.std(costs),np.mean(times))
        for i in range(maxiter):
            t1 = time()
            self.optimize(debug=False, show_progress=False)
            t2 = time()
            costs[i] = self.cost_function(self.state)
            times[i] = t2 - t1
        return (np.min(costs),np.max(costs),np.std(costs),np.mean(times))
    
    
# --------------------------------------------------------------------------------------------------------------------+
    
    
# source : https://programstore.ir/wp-content/uploads/2019/07/WOA.pdf
class whale_optimization:
    
    name = 'Whale Optimization'
    
    def __init__(self, 
                 cost_function,
                 popsize=20,
                 maxsteps=100):
    
        import numpy as np
        
        self.cost_function = cost_function
        self.d = cost_function.d
        self.input_domain = cost_function.input_domain
        self.popsize = popsize
        self.maxsteps = maxsteps
        
        self.state = None
        self.costs = None
        
    def update_state(self, state): self.state = state
    def update_costs(self, costs): self.costs = costs

    def optimize(self,
                 show_progress=True,
                 debug=False):
    
        import numpy as np
        rng = np.random.default_rng()
        population = np.array([[rng.uniform(a,b) for a,b in self.input_domain] for i in range(self.popsize)])
        clip = lambda x: [max(min(x, b), a) for (a,b),x in zip(self.input_domain,x)]
    
        search_agent_scores = np.array([self.cost_function(whale) for whale in population])
        best_agent_idx = np.argmin(search_agent_scores)
        best_agent = np.array(population[best_agent_idx])
        best_score = self.cost_function(best_agent)
    
        pop_avg_scores = []
        states, costs = [], []

        if show_progress:
            from tqdm import tqdm
            for i in tqdm(range(self.maxsteps)):
                a = 2-2*i/self.maxsteps # linearly decreasing a from 2 to 0 over the course of i
                r = rng.uniform()
                A = 2*a*r-a
                C = 2*r
                l = 2*rng.uniform()-1
                p = rng.uniform()
        
                for j in np.setdiff1d(range(self.popsize),best_agent_idx): # for all agent except the best one
                    agent = np.array(population[j])
            
                # Bubble-net attacking method (exploitation phase) ---+
                    if p < 0.5: # shrinking encircling mechanism
                        D = np.abs(C*np.array(best_agent) - np.array(agent))
                        if np.abs(A) < 1:
                            population[j] = clip(best_agent - A*D)
                        else: # Search for prey (exploration phase) ---+
                            random_agent = rng.choice(population)
                            population[j] = clip(random_agent-A*D)
                    else: # spiral model
                        D_prime = np.abs(best_agent - agent)
                        b = 1 # constant for defining the shape of the logarithmic spiral
                        population[j] = clip(D_prime*np.exp(b*l) * np.cos(2*np.pi*l) * best_agent)

                    search_agent_scores = [self.cost_function(whale) for whale in population]
                    best_current_score = min(search_agent_scores)
                    if best_current_score < self.cost_function(best_agent):
                        best_agent_idx = np.argmin(search_agent_scores)
                        best_agent = np.array(population[best_agent_idx])
                        best_score = self.cost_function(best_agent)
                        if debug: print('best search_agent so far: ' + str(best_score) + '\n')
        
                pop_avg_scores.append(np.mean(search_agent_scores)) # current generation avg. fitness
                costs.append(best_score)                            # fitness of the best individual
                
                self.update_state(best_agent)
                self.update_costs(costs)
            return best_agent, self.cost_function(best_agent), costs
        
        for i in range(self.maxsteps):
            a = 2-2*i/self.maxsteps
            r = rng.uniform()
            A = 2*a*r-a
            C = 2*r
            l = 2*rng.uniform()-1
            p = rng.uniform()
            for j in np.setdiff1d(range(self.popsize),best_agent_idx):
                agent = np.array(population[j])
                if p < 0.5:
                    D = np.abs(C*np.array(best_agent) - np.array(agent))
                    if np.abs(A) < 1:
                        population[j] = clip(best_agent - A*D)
                        random_agent = rng.choice(population)
                        population[j] = clip(random_agent-A*D)
                else:
                    D_prime = np.abs(best_agent - agent)
                    b = 1
                    population[j] = clip(D_prime*np.exp(b*l) * np.cos(2*np.pi*l) * best_agent)
                search_agent_scores = [self.cost_function(whale) for whale in population]
                best_current_score = min(search_agent_scores)
                if best_current_score < self.cost_function(best_agent):
                    best_agent_idx = np.argmin(search_agent_scores)
                    best_agent = np.array(population[best_agent_idx])
                    best_score = self.cost_function(best_agent)
                    if debug: print('best search agent so far: ' + str(best_score) + '\n')
            pop_avg_scores.append(np.mean(search_agent_scores))
            costs.append(best_score)
            self.update_state(best_agent)
            self.update_costs(costs)
        return best_agent, self.cost_function(best_agent), costs
    
    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(range(self.maxsteps),self.costs)
        plt.xlabel('iterations')
        plt.ylabel('cost')
        plt.show()
        
    def evaluate(self,
                 maxiter=30,
                 show_progress=True):
        
        import numpy as np
        from time import time
        costs = np.zeros(maxiter)
        times = np.zeros(maxiter)
        if show_progress:
            from tqdm import tqdm
            for i in tqdm(range(maxiter)):
                t1 = time()
                self.optimize(debug=False, show_progress=False)
                t2 = time()
                costs[i] = self.cost_function(self.state)
                times[i] = t2 - t1
            return (np.min(costs),np.max(costs),np.std(costs),np.mean(times))
        for i in range(maxiter):
            t1 = time()
            self.optimize(debug=False, show_progress=False)
            t2 = time()
            costs[i] = self.cost_function(self.state)
            times[i] = t2 - t1
        return (np.min(costs),np.max(costs),np.std(costs),np.mean(times))
    
        
# --------------------------------------------------------------------------------------------------------------------+
    
    
# source https://www.sciencedirect.com/science/article/pii/S0950705119305295
class equilibrium_optimizer:
    
    name = 'Equilibrium optimizer'
    
    def __init__(self, 
                 cost_function,
                 popsize=20,
                 maxsteps=100):
    
        import numpy as np
        
        self.cost_function = cost_function
        self.d = cost_function.d
        self.input_domain = cost_function.input_domain
        self.popsize = popsize
        self.maxsteps = maxsteps
        
        self.state = None
        self.costs = None
    
    def update_state(self, state): self.state = state
    def update_costs(self, costs): self.costs = costs
        
    def optimize(self,
                 show_progress=True,
                 debug=False):
    
        import numpy as np
        rng = np.random.default_rng()
        particles = np.array([[rng.uniform(a,b) for a,b in self.input_domain] for i in range(self.popsize)])
        a1, a2, GP, V = 2, 1, .5, 1
        clip = lambda x: [max(min(x, b), a) for (a,b),x in zip(self.input_domain,x)]
    
    
        # selection of the 4 particles with the smallest fitness value
        particles_concentrations = np.array([self.cost_function(particle) for particle in particles])
        #smallest_4particles_idxs = np.argpartition(particles_concentrations, 4)[:4]
        #C_eq1, C_eq2, C_eq3, C_eq4 = (particles[smallest_4particles_idxs[0]],
         #                             particles[smallest_4particles_idxs[1]],
          #                            particles[smallest_4particles_idxs[2]],
           #                           particles[smallest_4particles_idxs[3]])
    
    
        particles = np.array(sorted(particles, key = self.cost_function))
        C_eq1, C_eq2, C_eq3, C_eq4 = particles[: 4]
        states, costs = [], []
    
        # equilibrium pool constuction 
        C_av = np.mean((C_eq1, C_eq2, C_eq3, C_eq4), axis=0)
        C_eq_pool = {tuple(C_eq1), tuple(C_eq2), tuple(C_eq3), tuple(C_eq4), tuple(C_av)}

        if show_progress:
            from tqdm import tqdm
            for iter in tqdm(range(self.maxsteps)):

                for particle in particles:
                    if self.cost_function(particle)<self.cost_function(C_eq1):
                        C_eq1 = particle
                    elif self.cost_function(particle)>self.cost_function(C_eq1) and self.cost_function(particle)<self.cost_function(C_eq2):
                        C_eq2 = particle
                    elif self.cost_function(particle)>self.cost_function(C_eq1) and self.cost_function(particle)<self.cost_function(C_eq2) and self.cost_function(particle)<self.cost_function(C_eq3):
                        C_eq3 = particle
                    elif self.cost_function(particle)>self.cost_function(C_eq1) and self.cost_function(particle)<self.cost_function(C_eq2) and self.cost_function(particle)>self.cost_function(C_eq3) and self.cost_function(particles)<self.cost_function(C_eq4):
                        C_eq4 = particle          

                C_av = np.mean((C_eq1, C_eq2, C_eq3, C_eq4), axis=0)
                C_eq_pool = {tuple(C_eq1), tuple(C_eq2), tuple(C_eq3), tuple(C_eq4), tuple(C_av)}
                costs.append(self.cost_function(C_eq1))
        
#-------------Memory saving------------------+
                if iter==0: particles_old, particles_concentrations_old = particles, particles_concentrations
    
                for i in range(self.popsize):
                    if particles_concentrations_old[i] < particles_concentrations[i]:
                        particles_concentrations[i] = particles_concentrations_old[i]
                        particles[i] = particles_old_old[i]

                particles_old, particles_concentrations_old = particles, particles_concentrations
#--------------------------------------------+

                t = (1-iter/self.maxsteps)**(a2-iter/self.maxsteps)               # (Eq 9)
        
                for i in range(self.popsize):
                    C_eq = rng.choice(list(C_eq_pool))
                    lbda = rng.uniform(size=self.d)               # Generate lbda, r, according to (Eq 11)
                    r = rng.uniform(size=self.d)
                    F = a1*np.sign(r - .5) * (np.exp(-lbda * t) - 1)
                    r1, r2 = rng.uniform(size=2)
                    GCP = .5*r1*int(r2>=GP)
                    G0 = GCP*(C_eq-lbda*particles[i])
                    G = G0*F
                    particles[i] = clip(C_eq + (particles[i]-C_eq)*F + (G/lbda*V)*(1-F)) # (Eq 16)
                    
                self.update_state(C_eq1)
                self.update_costs(costs)

            return C_eq1, self.cost_function(C_eq1), costs
        
        for iter in range(self.maxsteps):
            for particle in particles:
                if self.cost_function(particle)<self.cost_function(C_eq1):
                    C_eq1 = particle
                elif self.cost_function(particle)>self.cost_function(C_eq1) and self.cost_function(particle)<self.cost_function(C_eq2):
                    C_eq2 = particle
                elif self.cost_function(particle)>self.cost_function(C_eq1) and self.cost_function(particle)<self.cost_function(C_eq2) and self.cost_function(particle)<self.cost_function(C_eq3):
                    C_eq3 = particle
                elif self.cost_function(particle)>self.cost_function(C_eq1) and self.cost_function(particle)<self.cost_function(C_eq2) and self.cost_function(particle)>self.cost_function(C_eq3) and self.cost_function(particles)<self.cost_function(C_eq4):
                    C_eq4 = particle          
            C_av = np.mean((C_eq1, C_eq2, C_eq3, C_eq4), axis=0)
            C_eq_pool = {tuple(C_eq1), tuple(C_eq2), tuple(C_eq3), tuple(C_eq4), tuple(C_av)}
            costs.append(self.cost_function(C_eq1))
            if iter==0: particles_old, particles_concentrations_old = particles, particles_concentrations
            for i in range(self.popsize):
                if particles_concentrations_old[i] < particles_concentrations[i]:
                    particles_concentrations[i] = particles_concentrations_old[i]
                    particles[i] = particles_old_old[i]
            particles_old, particles_concentrations_old = particles, particles_concentrations
            t = (1-iter/self.maxsteps)**(a2-iter/self.maxsteps)
            for i in range(self.popsize):
                C_eq = rng.choice(list(C_eq_pool))
                lbda = rng.uniform(size=self.d)
                r = rng.uniform(size=self.d)
                F = a1*np.sign(r - .5) * (np.exp(-lbda * t) - 1)
                r1, r2 = rng.uniform(size=2)
                GCP = .5*r1*int(r2>=GP)
                G0 = GCP*(C_eq-lbda*particles[i])
                G = G0*F
                particles[i] = clip(C_eq + (particles[i]-C_eq)*F + (G/lbda*V)*(1-F))
            self.update_state(C_eq1)
            self.update_costs(costs)
        return C_eq1, self.cost_function(C_eq1), costs
        
    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(range(self.maxsteps),self.costs)
        plt.xlabel('iterations')
        plt.ylabel('cost')
        plt.show()
        
    def evaluate(self,
                 maxiter=30,
                 show_progress=True):
        
        import numpy as np
        from time import time
        costs = np.zeros(maxiter)
        times = np.zeros(maxiter)
        if show_progress:
            from tqdm import tqdm
            for i in tqdm(range(maxiter)):
                t1 = time()
                self.optimize(debug=False, show_progress=False)
                t2 = time()
                costs[i] = self.cost_function(self.state)
                times[i] = t2 - t1
            return (np.min(costs),np.max(costs),np.std(costs),np.mean(times))
        for i in range(maxiter):
            t1 = time()
            self.optimize(debug=False, show_progress=False)
            t2 = time()
            costs[i] = self.cost_function(self.state)
            times[i] = t2 - t1
        return (np.min(costs),np.max(costs),np.std(costs),np.mean(times))

# source : https://en.wikiversity.org/wiki/Algorithm_models/Grey_Wolf_Optimizer
class grey_wolf_optimizer:

    name = 'Grey Wolf Optimizer'
    
    def __init__(self, 
                 cost_function,
                 popsize=20,
                 maxsteps=100):
    
        import numpy as np
        
        self.cost_function = cost_function
        self.d = cost_function.d
        self.input_domain = cost_function.input_domain
        self.popsize = popsize
        self.maxsteps = maxsteps
        
        self.state = None
        self.costs = None
    
    def update_state(self, state): self.state = state
    def update_costs(self, costs): self.costs = costs
        
    def optimize(self,
                 show_progress=True,
                 debug=False):
    
        import numpy as np
        rng = np.random.default_rng()
        pack = np.array([[rng.uniform(a,b) for a,b in self.input_domain] for i in range(self.popsize)])
        clip = lambda x: [max(min(x, b), a) for (a,b),x in zip(self.input_domain,x)]
    
        
        pack = np.array(sorted(pack, key = self.cost_function))
 
        # selection of the 3 wolves with the smallest fitness value
        alpha_wolf, beta_wolf, delta_wolf = pack[: 3]
        
        states, costs = [], []

        if show_progress:
            from tqdm import tqdm
            for iter in tqdm(range(self.maxsteps)):
                r = rng.uniform(size=self.d)
                a = 2*(1 - iter/self.maxsteps)
                for idx in range(self.popsize):
                    wolf = pack[idx]
                    
                    A1 = a*(2*rng.uniform()-1)
                    A2 = a*(2*rng.uniform()-1)
                    A3 = a*(2*rng.uniform()-1)
                    
                    C1, C2, C3 = 2*rng.uniform(), 2*rng.uniform(), 2*rng.uniform()
                    
                    D_alpha = np.abs(C1 * alpha_wolf - wolf)
                    D_beta = np.abs(C2 * beta_wolf - wolf)
                    D_delta = np.abs(C3 * delta_wolf - wolf)
                    
                    X1 = alpha_wolf - A1*D_alpha
                    X2 = beta_wolf - A2*D_beta
                    X3 = delta_wolf - A3*D_delta
                    
                    pack[idx] = np.mean((X1,X2,X3), axis=0)
                
                pack = np.array(sorted(pack, key = self.cost_function))
                alpha_wolf, beta_wolf, delta_wolf = pack[: 3]
                costs.append(self.cost_function(alpha_wolf))
                    
            self.update_state(alpha_wolf)
            self.update_costs(costs)

            return alpha_wolf, self.cost_function(alpha_wolf), costs
        
        for iter in range(self.maxsteps):
            r = rng.uniform(size=self.d)
            a = 2*(1 - iter/self.maxsteps)
            for idx in range(self.popsize):
                wolf = pack[idx]
                A1 = a*(2*rng.uniform()-1)
                A2 = a*(2*rng.uniform()-1)
                A3 = a*(2*rng.uniform()-1)
                C1, C2, C3 = 2*rng.uniform(), 2*rng.uniform(), 2*rng.uniform()
                D_alpha = np.abs(C1 * alpha_wolf - wolf)
                D_beta = np.abs(C2 * beta_wolf - wolf)
                D_delta = np.abs(C3 * delta_wolf - wolf)
                X1 = alpha_wolf - A1*D_alpha
                X2 = beta_wolf - A2*D_beta
                X3 = delta_wolf - A3*D_delta
                pack[idx] = np.mean((X1,X2,X3), axis=0)
            pack = np.array(sorted(pack, key = self.cost_function))
            alpha_wolf, beta_wolf, delta_wolf = pack[: 3]
            costs.append(self.cost_function(alpha_wolf))
        self.update_state(alpha_wolf)
        self.update_costs(costs)
        return alpha_wolf, self.cost_function(alpha_wolf), costs
        
    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(range(self.maxsteps),self.costs)
        plt.xlabel('iterations')
        plt.ylabel('cost')
        plt.show()

    
    def evaluate(self,
                 maxiter=30,
                 show_progress=True):
        
        import numpy as np
        from time import time
        costs = np.zeros(maxiter)
        times = np.zeros(maxiter)
        if show_progress:
            from tqdm import tqdm
            for i in tqdm(range(maxiter)):
                t1 = time()
                self.optimize(debug=False, show_progress=False)
                t2 = time()
                costs[i] = self.cost_function(self.state)
                times[i] = t2 - t1
            return (np.min(costs),np.max(costs),np.std(costs),np.mean(times))
        for i in range(maxiter):
            t1 = time()
            self.optimize(debug=False, show_progress=False)
            t2 = time()
            costs[i] = self.cost_function(self.state)
            times[i] = t2 - t1
        return (np.min(costs),np.max(costs),np.std(costs),np.mean(times))
    