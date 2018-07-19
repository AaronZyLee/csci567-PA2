from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):

    S = len(pi)
    N = len(O)
    delta = np.zeros([S, N])
    dp = pi
    for i in range(0,N):
        delta[:,i] = B[:,O[i]]*((A.T*dp).sum(axis=1))
        dp = delta[:,i]

    return delta


def backward(pi, A, B, O):
  """
  Backward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - gamma: A numpy array gamma[j, t] = P(Z_t = s_j, x_t+1:x_T)
  """
  S = len(pi)
  N = len(O)
  gamma = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  ###################################################
  
  return gamma

def seqprob_forward(delta):

    prob = 0
    prob = delta[:,delta.shape[1]-1].sum()
    return prob


def seqprob_backward(gamma, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward algorithm

  Inputs:
  - gamma: A numpy array gamma: A numpy array gamma[j, t] = P(Z_t = s_j, x_t+1:x_T)
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  
  return prob

def viterbi(pi, A, B, O):
  """
  Viterbi algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - path: A list of the most likely hidden state path k* (in terms of the state index)
    argmax_k P(s_k1:s_kT | x_1:x_T)
  """
  path = []
  ###################################################
  # Q3.3 Edit here
  ###################################################
  S = len(pi)
  N = len(O)
  delta = np.zeros([S, N])
  dp = pi
  for i in range(0,N):
    temp = A.T*dp*B[:,O[i]]
    dp = temp.max(axis=1)
    delta[:,i] = temp.argmax(axis=1)
  print(delta)
  print(dp)
  path.append(dp.argmax())
  print(path)
  for i in range(0,N-1):
    state = int(path[0])
    #path.insert(0,delta[state,N-1-i])
    inx = delta[state,N-1-i]
    path.insert(0,int(inx))
  return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  delta = forward(pi, A, B, O)
  gamma = backward(pi, A, B, O)

  prob1 = seqprob_forward(delta)
  prob2 = seqprob_backward(gamma, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()