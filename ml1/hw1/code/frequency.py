from urllib.request import urlopen as get

url = 'https://www.gutenberg.org/files/20727/20727.txt'
with get(url) as response:
    text = response.read().decode('utf-8')

# print(text)
# puts all words to lowercase
text = text.lower()

# remove punctuation 
punctuation = ".,!?;:\"()[]{}<>-_—'*“”’"
for p in punctuation:
    text = text.replace(p, " ")

count = {}

# reads txt as a list and counts each word until the end
for word in text.split():
    # checks the length of the word and counts the words greater than 5 char
    if len(word) > 5:
        if word not in count:
            count[word] = 1
        else:
            count[word] += 1

# loop that gets the 5 most sorted words
for word in sorted(count, key = count.get, reverse = True)[:10]: 
    print(word, ":", count[word])