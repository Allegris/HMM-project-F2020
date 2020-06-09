import sys
import numpy as np
import re

np.set_printoptions(threshold=sys.maxsize)


# Codons
start_codons = ["ATG", "ATC", "ATA", "ATT", "GTG", "GTT", "TTG"]
stop_codons = ["TAG", "TAA", "TGA"]
rev_stop_codons = ["CTA", "TTA", "TCA"]
rev_start_codons = ["CAT", "AAT", "CAC", "CAA", "TAT", "CAG", "GAT"]

# Parse "NCR" to "012"
def parse_annotation_to_numbers(annotation):
    sym_to_num = {'N': 0, 'C': 1, 'R': 2}
    return [sym_to_num[symbol.upper()] for symbol in annotation]


"""
Input is a matrix and we then divide each entry with the sum of its row
This is used to get:
1) The A matrix from the number of transitions matrix
2) The phi matrix from the number of emissions matrix
"""
def get_probability_matrix(matrix):
    prob_matrix = matrix/matrix.sum(axis=1)[:,None]
    return prob_matrix


"""
Training by Counting
Returns transition matrix, A, and emission matrix, phi
"""
def TbC_43_state(genome, predicted_genome):
    no_of_transitions_matrix, no_of_emissions_matrix = count_transitions_and_emissions(genome, predicted_genome)
    # Get transition matrix, A
    A = get_probability_matrix(no_of_transitions_matrix)
    # Get emission matrix, phi
    phi = get_probability_matrix(no_of_emissions_matrix)

    #return A, phi 
    return np.round(A, decimals=5), np.round(phi, decimals=5)    

"""
Returns:
Matrix containing no. of transitions from state i to state j in this form:
 0 1 2 ... 43
0
1
2
.
.
.
43

Matrix containing no. of emissions from state i to symbol j in this form:
 A C G T
0
1
2
.
.
.
43
Where the entries are the number of transitions and emissions given the values on the axes (+1 since pseudocounting)
(State 0 is Noncoding, states 1-21 are Coding and states 22-42 are Reverse coding)
"""
def count_transitions_and_emissions(genome, predicted_genome):
    
    # Initialized matrices - only contains 0s and 1s 
    # 0 means "cannot happen", i.e. should not be counted 1 means "can happen", i.e. should be counted)
    trans_dict, emis_dict = initialize_trans_and_emis_dict()
    
    # Count NNs
    trans_dict[0][0] += len(re.findall('(?=NN)', predicted_genome))
    
    # Count emissions from Ns
    emis_indices = [m.start() for m in re.finditer("N", predicted_genome)]
    for i in emis_indices:
        emis_dict[0][genome[i]] += 1
    
    # Valid start- and reverse stop codons and their state indices
    codon_state_dict_1 = {"ATG": 1, "GTG": 4, "TTG": 7, "TTA": 22, "CTA": 25, "TCA": 28}
    
    # Valid stop- and reverse start codons and their state indices
    codon_state_dict_2 = {"TAG": 13, "TGA": 16, "TAA": 19, "CAT": 34, "CAC": 37, "CAA": 40}
    
    # Count NCCCs and NRRRs with chosen start- and reverse stop codons
    for triple in ["CCC", "RRR"]:
        
        # Find all indices of all occurences of NCCC / NRRR
        indices = [m.start() for m in re.finditer("N" + triple, predicted_genome)]

        # Iterate through the NCCC / NRRR 
        for i in indices:
            codon = genome[i+1: i+4] # Start- / reverse stop codon
            start = i+1 # Index of first symbol in codon 
            stop = predicted_genome.find("N", start) # index of first N after gene
            codon_2 = genome[stop-3: stop] # Stop- / reverse start codon
            
            # Check if valid start- or reverse stop codon and valid stop- or reverse start codon
            if codon in codon_state_dict_1.keys() and codon_2 in codon_state_dict_2.keys():
                coding = triple == "CCC" # Are we in coding area?
                first_state = 10 if coding else 31 # First state in internal coding or reverse coding
                mid_state = 11 if coding else 32 # Middle state in internal coding or reverse coding
                last_state = 12 if coding else 33 # Last state in internal coding or reverse coding
                
                # Update transition to start / reverse stop codon
                trans_dict[0][codon_state_dict_1[codon]] += 1
                
                # Update transition to stop / reverse start codon                
                trans_dict[last_state][codon_state_dict_2[codon_2]] += 1
                                
                # Update internal coding / reverse coding transitions
                internal_codons_states_string = predicted_genome[start+3: stop-3] # Internal codons state string, e.g. CCCCCCCCC
                no_of_int_codons = internal_codons_states_string.count(triple) # No. of. internal codons, e.g. 3
                internal_codons_emis_string = genome[start+3: stop-3] # Corresponding emissions, e.g. AAATTTGGG
                
                # We make no_of_int_codons - 1 transisions from last state to first state
                trans_dict[last_state][first_state] += no_of_int_codons - 1
                
                # Count emissions from internal codons
                internal_codons = [internal_codons_emis_string[i:i+3] for i in range(0, len(internal_codons_emis_string), 3)] # Split internal codon emis string in blocks of 3 (i.e. into codons), e.g. ["AAA", "TTT", "GGG"]

                # Increase emissions from the 3 internal codon states
                for int_codon in internal_codons:
                    emis_dict[first_state][int_codon[0]] += 1
                    emis_dict[mid_state][int_codon[1]] += 1
                    emis_dict[last_state][int_codon[2]] += 1
                    
            # If codon is not valid:
            # Remove this gene from genome and predicted genome, since we do not want to count it
            else:
                predicted_genome = predicted_genome[0: start] if stop == -1 else predicted_genome[0: start] + predicted_genome[stop:]
                genome = genome[0: start] if stop == -1 else genome[0: start] + genome[stop:]

    # Convert dicts to matrices (np arrays)
    transitions_matrix = np.array([[v[j] for j in list(range(43))] for k, v in trans_dict.items()])
    emission_matrix = np.array([[v[j] for j in ["A", "C", "G", "T"]] for k, v in emis_dict.items()])
    return transitions_matrix, emission_matrix
    

"""
Returns:
Initial transision dict and emission dict with pseudocounted values (i.e. all inner values in dict ("?") are either 0 or 1).
The ? are 0 if "not possible" transition/emission and "1" if possible transition/emission.
Keys corresponding to states with transition and/or emission probability of 100% only contain one 1 and otherwise 0s.

The forms of the transition dict and emission dict are:
trans_dict = {0: {0: ?, 1: ?, ..., 42: ?},
              1: {0: ?, 1: ?, ..., 42: ?},
              ...
              41: {0: ?, 1: ?, ..., 42: ?},
              42: {0: ?, 1: ?, ..., 42: ?}}

emis_dict = {0: {"A": ?, "C": ?, "G": ?, "T": ?},
             1: {"A": ?, "C": ?, "G": ?, "T": ?},
             ...
             41: {"A": ?, "C": ?, "G": ?, "T": ?},
             42: {"A": ?, "C": ?, "G": ?, "T": ?}}
"""
def initialize_trans_and_emis_dict():
    # We have symbols A, C, G, T and 43 states (indexed 0 to 42)
    symbols = ["A", "C", "G", "T"]
    states = list(range(0,43))
    
    # We are doing pseudo counting (values set to 0 are not counted, i.e. "cannot happen")   
    transitions_dict = { i : { j : 0 for j in states} for i in states}
    emissions_dict = { i : { j : 0 for j in symbols} for i in states}
    
    # Update possible transitions and emission to 1 ("can happen")
    
    # Transitions which are 100%
    known_transitions = [(1, 2), (2, 3), (3, 10), (4, 5), (5, 6), (6, 10), (7, 8), (8, 9), (9, 10), # start codons
                         (10, 11), (11, 12), # internal coding 
                         (13, 14), (14, 15), (15, 0), (16, 17), (17, 18), (18, 0), (19, 20), (20, 21), (21, 0), #stop codons
                         (22, 23), (23, 24), (24, 31), (25, 26), (26, 27), (27, 31), (28, 29), (29, 30), (30, 31), #reverse stop codons
                         (31, 32), (32, 33), # internal reverse coding
                         (34, 35), (35, 36), (36, 0), (37, 38), (38, 39), (39, 0), (40, 41), (41, 42), (42, 0)] # reverse start codons
    
    # Set these transitions to 1 (pseudocounting) - all other cells in each "from state"'s row is 0, resulting in a 100% probability
    for (i, j) in known_transitions:
        transitions_dict[i][j] = 1
    
    # Transitions for which we need to determine probability
    possible_transitions = [(0, 0), (0, 1), (0, 4), (0, 7), (0, 22), (0, 25), (0, 28), #transisions from non-coding
                            (12, 10), (12, 13), (12, 16), (12, 19), #transitions from coding
                            (33, 31), (33, 34), (33, 37), (33, 40)] #transitions from reverse coding
    # Set these transitions to 1 (pseudocounting)
    for (i, j) in possible_transitions:
        transitions_dict[i][j] = 1
    
    # States for which emissions are given, i.e. 100% for some symbol
    known_emis_states = [[1, 2, 3], [4, 5, 6], [7, 8, 9], #start codon states
                         [13, 14, 15], [16, 17, 18], [19, 20, 21], #stop codon states
                         [22, 23, 24], [25, 26, 27], [28, 29, 30], #reverse stop codon states
                         [34, 35, 36], [37, 38, 39], [40, 41, 42]] #reverse start codon states
    
    # Corresponding emission symbols - these are the start and stop codons
    known_emis_symbols = [["A", "T", "G"], ["G", "T", "G"], ["T", "T", "G"], #start codons
                          ["T", "A", "G"], ["T", "G", "A"], ["T", "A", "A"], #stop codons
                          ["T", "T", "A"], ["C", "T", "A"], ["T", "C", "A"], #reverse stop codons
                          ["C", "A", "T"], ["C", "A", "C"], ["C", "A", "A"]] #reverse start codons
    
    # We combine the states and emissions into a list of form [(1, 'A'), (2, 'T'), (3, 'G'),...]
    known_emissions = []
    for sta, sym in list(zip(known_emis_states, known_emis_symbols)):
        known_emissions += list(zip(sta, sym))

    # Set these emissions to 1 (pseudocounting) - all other cells in each "emission state"'s row is 0, resulting in a 100% probability
    for (i, j) in known_emissions:
        emissions_dict[i][j] = 1
    
    # States for which we need to determine emission probabilities
    possible_emis_states = [0, 10, 11, 12, 31, 32, 33]
    
    # Emissions for which we need to determine probability
    possible_emissions = [(i, symbol) for i in possible_emis_states for symbol in symbols]
    
    # Set these emissions to 1 (pseudocounting)
    for (i, j) in possible_emissions:
        emissions_dict[i][j] = 1

    return transitions_dict, emissions_dict

    
    
    
    
# Code to run    
"""   
#count_transitions_and_emissions("CCATGAAAAAAAAATAGCC", "NNCCCCCCCCCCCCCCCNN")

A, phi = TbC_43_state("CCATGAAAAAAAAATAGCC", "NNCCCCCCCCCCCCCCCNN")
print(A)
print("\n***************************\n")
print(phi)
x = open("A.txt", "w")
x.write(str(A))
x.close()
"""
    
class hmm_43_state:
    def __init__(self, x, z):
        # Initial state probability vector, pi
        self.pi = np.zeros(43)
        self.pi[0] = 1 # We always start in state 0, i.e. N (noncoding)
        
        # Find transition matrix A and emission matrix phi by training by counting
        self.A, self.phi = TbC_43_state(x, z)
    
    
    
    
    
    