from nltk.stem import PorterStemmer

# no normalization
# no case change
# uppercasing
# lowercasing
# removing non-alphabet chars
# removing non-alphanumeric chars
# not removing word
# removing high frequent word > high_freq
# removing low frequent word < low_freq
# smoothing = alpha_num
def normalize_set(set_words):
    ps = PorterStemmer()

    popped = 0
    for i in range(len(set_words)):
        set_words[i-popped] = lowercase_word(set_words[i-popped])
        set_words[i-popped] = rm_nonalpha(set_words[i-popped])
        # set_words[i-popped] = ps.stem(set_words[i-popped])

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

def n_gram(n, set_words):

    if len(set_words) == n-1:
        return set_words
    else:
        for i in range(0, len(set_words)-(n-1), 1):
            word = ''
            
            for j in range(n):
                word += set_words[i+j]
                if j < n-1:
                    word += ' '

            set_words.append(word)