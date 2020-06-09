import numpy as np

'''
Parse "NCR" to "012"
'''
def parse_annotation_to_numbers(annotation):
    sym_to_num = {'N': 0, 'C': 1, 'R': 2}
    return [sym_to_num[symbol.upper()] for symbol in annotation]


'''
Input is a matrix and we then divide each entry with the sum of its row
This is used to get:
1) The A matrix from the number of transitions matrix
2) The phi matrix from the number of emissions matrix
'''
def get_probability_matrix(matrix):
    prob_matrix = matrix/matrix.sum(axis=1)[:,None]
    return prob_matrix


'''
Training by Counting
Returns transition matrix, A, and emission matrix, phi
'''
def TbC_7_state(genome, predicted_genome):
    no_of_transitions_matrix, no_of_emissions_matrix = count_transitions_and_emissions(genome, predicted_genome)
    # Get transition matrix, A
    A = get_probability_matrix(no_of_transitions_matrix)
    # Get emission matrix, phi
    phi = get_probability_matrix(no_of_emissions_matrix)

    #return A, phi 
    return np.round(A, decimals=5), np.round(phi, decimals=5)  


'''
Matrix containing no. of transitions from state i to state j in this form:
 0 1 2 3 4
0
1
2
3
4

Matrix containing no. of emissions from state i to symbol j in this form:
 A C G T
0
1
2
3
4
(State 0 is Noncoding and states 1-4 are Coding. Reverse coding is not modelled)
'''
def count_transitions_and_emissions(genome, predicted_genome):
    # Pseudo counting (values set to 0 are not counted, i.e. "cannot happen")
    transition_dict = {0: {0: 1, 1: 1, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0},
                       1: {0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0},
                       2: {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0},
                       3: {0: 1, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
                       4: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0},
                       5: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1},
                       6: {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0}}
    
    emission_dict = {0: {"A": 1, "C": 1, "G": 1, "T": 1},
                     1: {"A": 1, "C": 1, "G": 1, "T": 1},
                     2: {"A": 1, "C": 1, "G": 1, "T": 1},
                     3: {"A": 1, "C": 1, "G": 1, "T": 1},
                     4: {"A": 1, "C": 1, "G": 1, "T": 1},
                     5: {"A": 1, "C": 1, "G": 1, "T": 1},
                     6: {"A": 1, "C": 1, "G": 1, "T": 1}}

    # Parse "NCR" string to "012" string 
    predicted_genome_numbers = parse_annotation_to_numbers(predicted_genome)
    
    # Look at each pair of adjacent symbols (= each transition) in predicted_genome_numbers
    # Look at each pair of corresponding hidden state symbol and emission symbol (= each emission) in genome and predicted_genome
    i = 0
    #emission_dict[0][genome[0]] += 1 # we start in N and emit genome[0]
    while(i < (len(predicted_genome_numbers) - 1)):
        from_state = predicted_genome_numbers[i]
        to_state = predicted_genome_numbers[i+1]
        emit_symbol = genome[i]
        # NN
        if(from_state == 0 and to_state == 0):
            transition_dict[0][0] += 1
            emission_dict[0][emit_symbol] += 1
            i += 1
        # NC: count and jump over triplet
        elif(from_state == 0 and to_state == 1):
            transition_dict[0][1] += 1
            transition_dict[1][2] += 1
            transition_dict[2][3] += 1
            emission_dict[0][emit_symbol] += 1
            emission_dict[1][genome[i+1]] += 1
            emission_dict[2][genome[i+2]] += 1
            i += 3
        # CC
        elif(from_state == 1 and to_state == 1):
            transition_dict[3][1] += 1
            transition_dict[1][2] += 1
            transition_dict[2][3] += 1
            emission_dict[3][emit_symbol] += 1
            emission_dict[1][genome[i+1]] += 1
            emission_dict[2][genome[i+2]] += 1
            i += 3
        # CN
        elif(from_state == 1 and to_state == 0):
            transition_dict[3][0] += 1
            emission_dict[3][emit_symbol] += 1
            i += 1
        # NR: count and jump over triplet
        elif(from_state == 0 and to_state == 2):
            transition_dict[0][4] += 1
            transition_dict[4][5] += 1
            transition_dict[5][6] += 1
            emission_dict[0][emit_symbol] += 1
            emission_dict[4][genome[i+1]] += 1
            emission_dict[5][genome[i+2]] += 1
            i += 3
        # RR
        elif(from_state == 2 and to_state == 2):
            transition_dict[6][4] += 1
            transition_dict[4][5] += 1
            transition_dict[5][6] += 1
            emission_dict[6][emit_symbol] += 1
            emission_dict[4][genome[i+1]] += 1
            emission_dict[5][genome[i+2]] += 1
            i += 3
        # CN
        elif(from_state == 2 and to_state == 0):
            transition_dict[6][0] += 1
            emission_dict[6][emit_symbol] += 1
            i += 1
        else:
            i += 1
    transitions_matrix = np.array([[v[j] for j in [0, 1, 2, 3, 4, 5, 6]] for k, v in transition_dict.items()])
    emission_matrix = np.array([[v[j] for j in ["A", "C", "G", "T"]] for k, v in emission_dict.items()])
    return transitions_matrix, emission_matrix



class hmm_7_state:
    def __init__(self, x, z):
        # Initial state probability vector, pi
        self.pi = np.zeros(7)
        self.pi[0] = 1 # We always start in state 0, i.e. N (noncoding)
        
        # Find transition matrix A and emission matrix phi by training by counting
        self.A, self.phi = TbC_7_state(x, z)
        
        
    
    
