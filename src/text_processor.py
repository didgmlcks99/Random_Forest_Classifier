from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def setting_tokenizer(line, tk_case):
    if tk_case == True:
        # @Usairway hello = ['@', 'Usairway', 'hello']
        return word_tokenize(line)
    else:
        # @Usairway hello = ['@Usairway', 'hello']
        return line.split()

def normalize_set(set_words):
    ps = PorterStemmer()

    popped = 0
    for i in range(len(set_words)):
        set_words[i-popped] = lowercase_word(set_words[i-popped])
        set_words[i-popped] = rm_nonalpha(set_words[i-popped])
        set_words[i-popped] = ps.stem(set_words[i-popped])

        # emptied word
        if not set_words[i-popped]:
            set_words.pop(i-popped)
            popped+=1

def lowercase_word(word):
    return word.lower()

def uppercase_word(word):
    return word.upper()

def rm_nonalpha(word):
    return ''.join(filter(str.isalpha, word))
    
def rm_nonalnum(word):
    return ''.join(filter(str.isalnum, word))

def n_gram(set_words, gram_num):

    if len(set_words) == gram_num-1:
        return set_words
    else:
        for i in range(0, len(set_words)-(gram_num-1), 1):
            word = ''
            
            for j in range(gram_num):
                word += set_words[i+j]
                if j < gram_num-1:
                    word += ' '

            set_words.append(word)
