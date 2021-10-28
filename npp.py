# Copyright 2020 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## ------- import packages -------
from dwave.system import DWaveSampler, EmbeddingComposite

def get_token():
    '''Returns personal access token. Only required if submitting to autograder.'''
    
    my_token = "DEV-03efd410eb1fa5ac2dc9e46ae4ceb734f8598f54"
    return my_token

# TODO:  Add code here to define your QUBO dictionary
def get_qubo(S):
    """Returns a dictionary representing a QUBO.
    Args:
        S(list of integers): represents the numbers being partitioned
    """

    # define Q matrix Rigon Liu
    Q = {}

    print(f"\nS:  {S}")

    # get C
    C = 0
    for i in S:
        C += i

    # get Q diagonal elements
    for i in range(8):
        Q[(i,i)] = -4*C*S[i] + 4*S[i]*S[i]

    # get Q upper-triangular elements
    for i in range(8):
        for j in range(i+1,8):
            Q[(i,j)] = 8*S[i]*S[j]

    # get Q bottem-triangular elements
    for i in range(1,8):
        for j in range(0,i-1):
            Q[(i,j)] = 0
    
    print("\nQUBO:\n")
    for i in range(8):
        row = ''
        for j in range(8):
            if (i,j) in Q:
                row += str(Q[(i,j)])+'\t'
            else:
                row += str(0) + '\t'
        print(row)

    print("\n---------------------------------------------------------------------------------------------------------------- \n")

    return Q

# TODO:  Choose QPU parameters in the following function
def run_on_qpu(Q, sampler):
    """Runs the QUBO problem Q on the sampler provided.
    Args:
        Q(dict): a representation of a QUBO
        sampler(dimod.Sampler): a sampler that uses the QPU
    """

    chainstrength = 20000 # update
    numruns = 100 # update

    print(f"chain_strength = {chainstrength}")
    print(f"number of runs = {numruns}\n")
    
    sample_set = sampler.sample_qubo(Q, chain_strength=chainstrength, num_reads=numruns, label='Training - Number Partitioning')

    return sample_set


## ------- Main program -------
if __name__ == "__main__":

    ## ------- Set up our list of numbers -------
    S = [25, 7, 13, 31, 42, 17, 21, 10]

    ## ------- Set up our QUBO dictionary -------

    Q = get_qubo(S)

    ## ------- Run our QUBO on the QPU -------

    sampler = EmbeddingComposite(DWaveSampler())

    sample_set = run_on_qpu(Q, sampler)

    ## ------- Return results to user -------
    for sample in sample_set:
        S1 = [S[i] for i in sample if sample[i] == 1]
        S0 = [S[i] for i in sample if sample[i] == 0]
        print("S0 Sum: ", sum(S0), "\tS1 Sum: ", sum(S1), "\t", S0)

    
