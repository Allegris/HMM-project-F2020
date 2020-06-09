import numpy as np
from Bio import SeqIO
import os
import hmm_3_state as hmm3
import hmm_5_state as hmm5
import hmm_7_state as hmm7
import hmm_43_state as hmm43

# Read fasta files
def read_fasta_file(filename):
    for record in SeqIO.parse(filename, "fasta"):  
        return record.seq

'''
Translates a prediction string of indices [0, 1, 2]to a prediction of letters [N, C, R]
Input: String of form "000011111111100000222222222222000" and 
Returns: The correct string of letters [0, 1, 2] for the HMM with the given number of states
'''
def translate_state_indices_to_NCR(predicted_genome, no_of_states):
    prediction = ""
    if no_of_states == 3:
        for i in predicted_genome:
            if i == 0:
                prediction += "N"
            elif i == 1:
                prediction += "C"
            elif i == 2:
                prediction += "R"
    elif no_of_states == 5:
        for i in predicted_genome:
            if i == 0:
                prediction += "N"
            else:
                prediction += "C"
    elif no_of_states == 7:
        for i in predicted_genome:
            if i == 0:
                prediction += "N"
            if 1 <= i and i <= 3:
                prediction += "C"
            if 4 <= i and i <= 6:
                prediction += "R"
    elif no_of_states == 43:
        for i in predicted_genome:
            if i == 0:
                prediction += "N"
            if 1 <= i and i <= 21:
                prediction += "C"
            if 22 <= i and i <= 42:
                prediction += "R"
    return prediction


"""
Viterbi algorithm
Fills out the omega table in log space (i.e. omega hat)
"""
def compute_viterbi_table(hmm, genome):
    k = hmm.phi.shape[0]
    n = len(genome)
    omega_hat = np.zeros((k, n))
    
    # Get the column index corresponding to an emission symbol
    symbol_to_column_dict = {"A": 0, "C": 1, "G": 2, "T": 3}

    #Fill out first column of table
    first_emis_col = symbol_to_column_dict[genome[0]]
    omega_hat[:, 0] = np.log(hmm.pi) + np.log(hmm.phi[:, first_emis_col])

    # Fill out table column by column
    for j in range(1, n):
        if j % 10000 == 0:
            print("omega:", j * 100 / n, "%")
        symbol_col = symbol_to_column_dict[genome[j]]
        omega_hat[:, j] = np.log(hmm.phi[:, symbol_col]) + [np.max(omega_hat[:, j-1] + np.log(hmm.A[:, i])) for i in range(k)]     
    
    return np.array(omega_hat)


def backtrack_viterbi(hmm, genome, omega_hat):
    k = hmm.phi.shape[0]
    n = len(genome)
    z = np.zeros(n, dtype=int)

    # Get the column index corresponding to an emission symbol
    symbol_to_column_dict = {"A": 0, "C": 1, "G": 2, "T": 3}

    # Find largest value in last column of omega hat table
    z[n - 1] = np.argmax(omega_hat[:, n - 1])

    for j in reversed(range(0, n - 1)):
        symbol_col = symbol_to_column_dict[genome[j+1]]
        z[j] = np.argmax(np.log(hmm.phi[z[j+1], symbol_col]) + [omega_hat[i, j] + np.log(hmm.A[i, z[j+1]]) for i in range(k)])
    return z


def predict(hmm, genome, genome_name, seq_name, no_of_states):
    om_hat = compute_viterbi_table(hmm, genome)
    z = backtrack_viterbi(hmm, genome, om_hat)
    z = translate_state_indices_to_NCR(z, no_of_states)
    path = 'predictions/'
    if not os.path.exists(path):
        os.makedirs(path)
    fasta = open(path + str(no_of_states) + "_state_prediction_of_" + genome_name + ".fa", "w+")
    fasta.write(">" + seq_name + "\n" + z)
    fasta.close()
    
    
"""
Tests that a HMM is valid, i.e. that pi sums to 1, and the rows in A and phi sum to 1
and that they all only contain numbers in range 0 to 1 (probabilities)
"""
def validate_hmm(model):
    if not np.allclose(np.sum(model.pi), 1):
        return False, "Initial probabilities do not sum to 1"
    if not np.allclose(np.sum(model.A, axis=1), 1):
        return False, "Transition probabilities do not sum to 1, but to "
    if not np.allclose(np.sum(model.phi, axis=1), 1):
        return False, "Emission probabilities do not sum to 1"
    for number in model.pi:
        if number < 0 or number > 1:
            return False, "An initial probability is not in range 0 to 1"
    for line in model.A:
        for number in line:
            if number < 0 or number > 1:
                return False, "A transition probability is not in range 0 to 1"
    for line in model.phi:
        for number in line:
            if number < 0 or number > 1:
                return False, "An emission probability is not in range 0 to 1"
    return True


'''
Input: 
    Filename of known genome, known_genome
    Filename of known annotated genome, known_ann_genome
    Filename of genome to predict, genome_to_predict
    Number of states in the HMM to run (i.e. 3, 5 or 43)
'''
def read_files_and_predict(known_genome_file, known_ann_genome_file, genome_to_predict_file, no_of_states):
    # Read known genome files
    known_genome_file = known_genome_file + ".fa"
    known_ann_genome_file = known_ann_genome_file + ".fa"
    known_genome = read_fasta_file(known_genome_file)
    known_ann_genome = read_fasta_file(known_ann_genome_file)
    # Create HMM
    hmm = None
    if no_of_states == 3:
        hmm = hmm3.hmm_3_state(known_genome, known_ann_genome)
    elif no_of_states == 5:
        hmm = hmm5.hmm_5_state(known_genome, known_ann_genome)
    elif no_of_states == 7:
        hmm = hmm7.hmm_7_state(known_genome, known_ann_genome)
    elif no_of_states == 43:
        hmm = hmm43.hmm_43_state(str(known_genome), str(known_ann_genome))
    print("pi", hmm.pi)
    print("A", hmm.A)
    print("phi", hmm.phi)
    # Validate
    print("Validation of HMM (should be True): ", validate_hmm(hmm))
    # Predict
    genome_to_predict_file = genome_to_predict_file + ".fa"
    genome_to_predict = read_fasta_file(genome_to_predict_file)
    predict(hmm, genome_to_predict, genome_to_predict_file, "pred-ann10", no_of_states)

'''
Reads and predicts on given files with HMMs of given number of states
'''
def run_tests(list_no_of_states):
    for no_of_states in list_no_of_states:
        read_files_and_predict("genome1", "true-ann1", "genome10", no_of_states)
    

        
'''
Code to run
'''
no_of_states_HMMs = [7] #[3, 5, 7, 43] 
run_tests(no_of_states_HMMs)





