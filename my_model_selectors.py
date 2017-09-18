import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
	""" select the model with the lowest Bayesian Information Criterion(BIC) score

	http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
	Bayesian information criteria: BIC = -2 * logL + p * logN
	"""

	def select(self):
		""" select the best model for self.this_word based on
		BIC score for n between self.min_n_components and self.max_n_components

		:return: GaussianHMM object
		"""
		warnings.filterwarnings("ignore", category=DeprecationWarning)

		# Initialize variables
		best_bic_score = float("-inf")
		best_model = None
		N = sum(self.lengths)

		# Loop through model parameters to initialize model
		for n_states in range(self.min_n_components, self.max_n_components+1):

			# Initialize the model
			hmm_model = self.base_model(n_states)

			try:
				# Get the log score
				logL = hmm_model.score(self.X, self.lengths)

			except:
				continue

			# Determine the model performance
			n = hmm_model.n_components # number of data points
			d = hmm_model.n_features # number of features
			p = n**2 + 2 * n * d - 1 # number of parameters
			bic_score = -2*logL + p*np.log(N) # bic equation

			# Check the performance against best model so far
			if bic_score > best_bic_score:
				best_bic_score = bic_score
				best_model = hmm_model

		# Return best model
		return best_model


class SelectorDIC(ModelSelector):
	''' select best model based on Discriminative Information Criterion

	Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
	Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
	http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
	https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
	DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
	'''

	def select(self):
		warnings.filterwarnings("ignore", category=DeprecationWarning)

		# Initialize variables
		best_dic_score = float("-inf")
		best_model = None

		# Loop through model parameters to initialize model
		for n_states in range(self.min_n_components, self.max_n_components+1):

			# Initialize the model
			hmm_model = self.base_model(n_states)

			try:
				# Score the model performance
				logL = hmm_model.score(self.X, self.lengths)

			except:
				continue

			sumLog = 0.0
			total = 0

			# Loop through words
			for word in self.words.keys():

				total += 1

				if word != self.this_word:

					# Logs summation
					X, lengths = self.hwords[word]
					sumLog += hmm_model.score(X, lengths)

			# Calculate average log
			avgLog = sumLog/total

			# dic equation
			dic_score = logL + avgLog

			# Check the performance against best model so far
			if dic_score > best_dic_score:
				best_dic_score = dic_score
				best_model = hmm_model

		# Return best model
		return best_model


class SelectorCV(ModelSelector):
	''' select best model based on average log Likelihood of cross-validation folds

    '''

	def select(self):
		warnings.filterwarnings("ignore", category=DeprecationWarning)

		# Initialize KFold, model, and log scores
		best_logL = float("-inf")
		n_splits = min(len(self.lengths), 3)
		split_method = KFold(n_splits)

		# Loop through model parameters to initialize model
		for n_states in range(self.min_n_components, self.max_n_components+1):

			# Intialize total log score
			total_logL = 0

			# Loop through k-fold cross validation sets
			for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

				# Initialize the training and testing data
				X_train, length_train = combine_sequences(cv_train_idx, self.sequences)
				X_test, length_test = combine_sequences(cv_test_idx, self.sequences)

				try:
					# Initialize the model
					hmm_model = GaussianHMM(n_components=n_states, n_iter=1000).fit(X_train, length_train)

					# Score the model performance
					logL = hmm_model.score(X_test, length_test)

					# Sum up the total log score
					total_logL += logL

				except:
					pass

			# Calculate the average log score
			avg_logL = total_logL/n_splits

			# Check the performance against best model so far
			if avg_logL > best_logL:
				best_logL = avg_logL
				best_model = self.base_model(n_states)

		# Return best model
		return best_model

