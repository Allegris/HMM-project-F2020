import numpy as np



'''
Training by Counting
Returns transition matrix, A, and emission matrix, phi
'''
def TbC_3_state(genome, predicted_genome):
    # Count transitions and emissions
    trans_matrix, emis_matrix = count_transitions_and_emissions(genome, predicted_genome)
    
    # Get transition matrix, A, and emission matrix, phi
    # We divide each entry by the row sum
    A = trans_matrix/trans_matrix.sum(axis=1)[:,None] 
    phi = emis_matrix/emis_matrix.sum(axis=1)[:,None]
    
    # Return matrices, rounded to 5 decimals
    return np.round(A, decimals=5), np.round(phi, decimals=5)    



'''
Returns two matrices (numpy arrays): 

Matrix containing no. of transitions from state i to state j in this form:
 N C R
N
C
R

and 

Matrix containing no. of emissions from state i to symbol j in this form:
 A C G T
N
C
R
'''
def count_transitions_and_emissions(genome, predicted_genome):
    # Pseudo-counting number of transitions and emissions
    transition_dict = {"N": {"N": 1, "C": 1, "R": 1},
                       "C": {"N": 1, "C": 1, "R": 0},
                       "R": {"N": 1, "C": 0, "R": 1}}

    emission_dict = {"N": {"A": 1, "C": 1, "G": 1, "T": 1},
                     "C": {"A": 1, "C": 1, "G": 1, "T": 1},
                     "R": {"A": 1, "C": 1, "G": 1, "T": 1}}
    
    # Look at each pair of adjacent symbols (= each transition) in predicted_genome and
    # Look at each pair of corresponding hidden state symbol and emission symbol 
    # (= each emission) in genome and predicted_genome
    for i in range(len(predicted_genome)-1):
        from_state = predicted_genome[i]
        to_state = predicted_genome[i+1]
        both_states = from_state + to_state
        # Count transitions
        if both_states != "CR" and both_states != "RC":
            transition_dict[from_state][to_state] += 1
        #Count emissions
        emission_symbol = genome[i]
        emission_dict[from_state][emission_symbol] += 1
    
    # Make dicts into matrices (numpy arrays) and return them
    transitions_matrix = np.array([[v[j] for j in ["N", "C", "R"]] for k, v in transition_dict.items()])
    emission_matrix = np.array([[v[j] for j in ["A", "C", "G", "T"]] for k, v in emission_dict.items()])
    return transitions_matrix, emission_matrix


'''
HMM 3 state class with init function
'''
class hmm_3_state:
    def __init__(self, x, z):
        # Initial state probability vector, pi
        self.pi = np.zeros(3)
        self.pi[0] = 1 # We always start in state 0, i.e. N (noncoding)
        # Find transition matrix A and emission matrix phi by training by counting
        self.A, self.phi = TbC_3_state(x, z)
        
        
    
    
