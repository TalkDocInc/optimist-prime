import re
word_list = []
from math import prod

with open('/usr/share/dict/words', 'r') as f:
    # Read the file and split into a list of words
    word_list = f.read().split()

class PrimeComplexityEngine:
    def __init__(self):
        # Alphabet prime mapping
        self.primes_map = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
        self.char_map = {chr(i + 97): self.primes_map[i] for i in range(26)}
        self.baseline_lambda = 0.15

    def get_word_prime_value(self, word: str) -> int:
        clean_word = re.sub(r'[^a-z]', '', word.lower())
        return prod(self.char_map.get(c, 0) for c in clean_word)


from random import randint

def words_to_use_in_response(input_string, m_samples, persons_name):
    PCE = PrimeComplexityEngine()
    
    input_words = input_string.split(" ")
    output_words = []

    for i in range(0, len(input_words), m_samples):
        sum_so_far = PCE.get_word_prime_value(persons_name) + randint(0,1000)
        for q in range(i, min(len(input_words),i+m_samples)):
            val = PCE.get_word_prime_value(input_words[q])
            sum_so_far += val
        
        output_words.append(word_list[sum_so_far % len(word_list)])
    
    return output_words

def words_to_use_in_outgoing_messages_or_program_to_collect_advice_or_program_to_take_action_in_reality(context, m_samples, persons_name, random=False):
    PCE = PrimeComplexityEngine()
    input_string = context
    input_words = input_string.split(" ")
    output_words = []


    for i in range(0, len(input_words), m_samples):
        sum_so_far = PCE.get_word_prime_value(persons_name)
        if random:
            sum_so_far += randint(0,1000)

        for q in range(i, min(len(input_words),i+m_samples)):
            val = PCE.get_word_prime_value(input_words[q])
            sum_so_far += val

        output_words.append(word_list[sum_so_far % len(word_list)])

    return output_words



    
