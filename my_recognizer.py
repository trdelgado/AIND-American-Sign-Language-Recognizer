import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # TODO implement the recognizer

    # Loop through test sequence
    test_sequence = list(test_set.get_all_Xlengths().values())
    for test_X, test_Xlength in test_sequence:

      # Initialize probability dictionary
      probability = {}

      # Loop through each word, model pair
      for word, model in models.items():

        # Calculate the log score for each word model
        try:
            logL = model.score(test_X, test_Xlength)
        except:
            logL = float("-inf")

        # Store word, log score pair
        probability[word] = logL
      
      # Update the proabilities list
      probabilities.append(probability)

      # Determine the max score for each model
      # Took inspiration for the Udacity discussion board to find the max score:
      # https://discussions.udacity.com/t/recognizer-implementation/234793/40
      guess = max([(logL, word) for word, logL in probability.items()])[1]

      # Update the tested word 
      guesses.append(guess)

    # return probabilities, guesses
    return (probabilities, guesses)
