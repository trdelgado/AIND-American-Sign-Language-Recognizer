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

    # Loop through each word id
    for word_id in range(len(test_set.get_all_Xlengths())):

      # Initialize data and probability dictionary
      current_sequence = test_set.get_item_sequences(word_id)
      current_length = test_set.get_item_Xlengths(word_id)
      probability = {}

      # Loop through each word, model pair
      for word, model in models.items():

        # Calculate the log score for each word model
        try:
            logL = model.score(current_sequence, current_length)
        
        except:
            logL = float("-inf")

        # Store word, log score pair
        probability[word] = logL
      
      # Update the proabilities list
      probabilities.append(probability)

      # Determine the max score for each model
      guess = max([(v,k) for k, v in probability.items()])[1]

      # Update the tested word 
      guesses.append(guess)

    # return probabilities, guesses
    return (probabilities, guesses)
