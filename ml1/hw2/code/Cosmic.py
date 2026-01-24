# question 2
from urllib.request import urlopen as get
from matplotlib import pyplot as plt
import numpy as np
import string

url = 'https://www.gutenberg.org/files/20727/20727.txt'
with get(url) as response:
    text = response.read().decode('utf-8')

# coverts characters in txt to uppercase and discards punctuation
character = []
for char in text:
    if char.isalpha():
        character.append(char.upper())

# Counts each character 
count = {}
for char in character:
    if char not in count:
        count.update({char: 1})
    else: 
        count[char] += 1

# sorts the dictionary 
count = dict(sorted(count.items(), key=lambda x: x[1], reverse=True))

# create array of letters and occurances of each one
x = []
y = []
# maps frequency of each letter and letter to their respective arrays
for letter in string.ascii_uppercase:
    x.append(letter)
    y.append(count[letter])

# sorting x y arrays for the plot
xy = sorted(zip(x, y), key=lambda pair: pair[1])
x, y = zip(*xy)

plt.bar(x, y)
plt.title('Character Frequency in "The Cosmic Computer" by Piper')
plt.xlabel("Alphabet")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

plt.show()