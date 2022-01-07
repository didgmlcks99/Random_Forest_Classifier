import re

str = "hello my name is heechan-yang"

words = str.split()

print(words)

for i in range(words):
    splitted = re.split('[^a-zA-Z]', words[i])

    # if len(splitted) > 1:


    print(splitted)