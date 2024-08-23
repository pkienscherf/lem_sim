import numpy as np
import pandas as pd
import os

from lem_sim import communication
from lem_sim import client
from lem_sim import contract
from lem_sim import linearoptimization as lp

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "data")

''' Read in and process problem data '''
D_df = pd.read_csv(os.path.join(data_dir, "D.csv")) # valuation data (d)
PTDF_df = pd.read_csv(os.path.join(data_dir, "PTDF.csv")) # PTDF matrix (here 109 x 55, later transposed)
Limits_df = pd.read_csv(os.path.join(data_dir, "Limits.csv")) # Line limits (109 x 1)
Loads_df = pd.read_csv(os.path.join(data_dir, "Loads_h.csv")).iloc[17:17+4] # non-EV loads, hours 17h to 21h - 4 time steps

T, N = Loads_df.shape
L = PTDF_df.shape[0]
Delta = 50. # energy needed.
Pmax = 11.

''' Definition of the Central Optimization Problem '''
TARGET_COEFS = -np.concatenate(np.array(D_df).transpose())  # cost vectors (d)
INDIVIDUAL_RESOURCES = np.tile(np.append(Delta, Pmax*np.ones(T)), N)  # individual resources (n)
INDIVIDUAL_COEFS = np.kron(np.eye(N), np.concatenate((np.ones(T).reshape(1,T), np.eye(T)), axis=0))  # individual coefficients(N)
SHARED_RESOURCES = (np.array(Limits_df) - np.matmul(np.array(PTDF_df), np.array(Loads_df).transpose())).flatten()  # shared resources (c)

SHARED_COEFS = np.zeros((T*L, T*N))
for t in range(T):
    SHARED_COEFS[t:(T*L):T, t:(T*N):T] = np.array(PTDF_df)

class Variables(object):

    def __init__(self):
        self._central_problem = lp.OptimizationProblem(TARGET_COEFS, INDIVIDUAL_RESOURCES, INDIVIDUAL_COEFS, SHARED_RESOURCES, SHARED_COEFS)
        self._web3 = communication.get_network_connection()
        self._dealer_contract = contract.ContractHandler(self._web3, 'Dealer.json')
        self._accounts = self._web3.eth.accounts
        self._dealer = client.Dealer(self._accounts.pop(0), self._web3, self._dealer_contract, self._central_problem.shared_resources.size)
        self._agent_pool = [client.Agent(number, account, self._web3, self._dealer_contract) for number, account in enumerate(self._accounts, 1)]
        self._amount_agents = len(self._agent_pool)
        self._latest_block = self._web3.eth.get_block('latest')['number']

    @property
    def web3(self):
        return self._web3

    @property
    def accounts(self):
        return self._accounts

    @property
    def agent_pool(self):
        return self._agent_pool

    @property
    def amount_agents(self):
        return self._amount_agents

    @property
    def dealer_contract(self):
        return self._dealer_contract

    @property
    def dealer(self):
        return self._dealer

    @property
    def central_problem(self):
        return self._central_problem
    
    @property
    def latest_block(self):
        return self._latest_block

    @latest_block.setter
    def latest_block(self, value):
        self._latest_block = value
