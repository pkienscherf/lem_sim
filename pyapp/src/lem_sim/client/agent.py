import numpy as np
from scipy.optimize import linprog


class Agent(object):

    def __init__(self, account_address, provider, dealer_contract):
        self._account_address = account_address
        self._provider = provider
        self._dealer_contract = dealer_contract
        self._optimization_problem = None
        self._bundle_set = None
        self._bid = None
        self._bill = None
        self._mkt_prices = None

    @property
    def account_address(self):
        return self._account_address

    @property
    def balance(self):
        return self._provider.eth.getBalance(self._account_address)

    @property
    def bundle_set(self):
        return self._bundle_set

    @bundle_set.setter
    def bundle_set(self, value):
        self._bundle_set = value

    @property
    def bid(self):
        return self._bid

    @bid.setter
    def bid(self, value):
        self._bid = value

    @property
    def optimization_problem(self):
        return self._optimization_problem

    @optimization_problem.setter
    def optimization_problem(self, value):
        self._optimization_problem = value
        self.determine_bundle_attributes()

    def send_transaction(self, receiver_account_address, value):
        self._provider.eth.sendTransaction({'to': receiver_account_address, 'from': self.account_address, 'value': value})

    def determine_bundle_attributes(self):
        result = solve_bundle_determination(self._optimization_problem)
        self._bundle_set = result.x
        self._bid = abs(self._optimization_problem.solve().fun - result.fun)

    def get_mkt_prices(self):
        return self._dealer_contract.contract.functions.getMktPrices().call()

    def get_bill(self):
        self._bill = self._dealer_contract.contract.functions.getBill().call({'from': self._account_address})

    def get_trade(self):
        trade = self._dealer_contract.contract.functions.getTrade().call({'from': self._account_address, 'value': self._bill})
        self._dealer_contract.contract.functions.getTrade().transact({'from': self._account_address, 'value': self._bill})
        return trade

    def set_order(self):
        self._dealer_contract.contract.functions.setOrder(self._bundle_set, self._bid).transact({'from': self._account_address})


def solve_bundle_determination(optimization_problem):
        bundle_size = optimization_problem.shared_resources.size

        bundle_target_coefs = np.concatenate((optimization_problem.target_coefs, np.zeros(bundle_size)))
        bundle_individual_coefs = np.concatenate((optimization_problem.individual_coefs, np.zeros(optimization_problem.individual_coefs.shape)), axis=1)
        bundle_shared_coefs = np.concatenate((optimization_problem.shared_coefs, np.identity(bundle_size, dtype=float) * (-1)), axis=1)

        bundle_coefs = np.concatenate((bundle_individual_coefs, bundle_shared_coefs))
        bundle_resources = np.concatenate((optimization_problem.individual_resources, optimization_problem.shared_resources))

        result = linprog(bundle_target_coefs, bundle_coefs, bundle_resources)
        result.x = result.x[-bundle_size:]  # remove not bundle coefficients

        return result
