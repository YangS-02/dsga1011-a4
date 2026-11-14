import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

RNG = random.Random(0)

def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

# ======================================================================================================================
# Variables
# ======================================================================================================================
# keyboard adjacent map
KEYBOARD_ADJACENT = {
    # First row
    'q': 'wa', 'w': 'qeasd', 'e': 'wrsdf', 'r': 'etdfg', 't': 'ryfgh', 'y': 'tughj', 'u': 'yihjk', 'i': 'uojkl', 'o': 'ipkl', 'p': 'ol',
    # Second row
    'a': 'qwsz', 's': 'weadzx', 'd': 'ersfxc', 'f': 'rtdgcv', 'g': 'tyfhvb', 'h': 'yugjbn', 'j': 'uihknm', 'k': 'iojlm', 'l': 'opk',
    # Last row
    'z': 'asx', 'x': 'sdzc', 'c': 'dfxv', 'v': 'fgcb', 'b': 'ghvn', 'n': 'hjbm', 'm': 'jkn'
}
# Some caveat: this adjacent mapping is rough, and we are sampling uniformly from the adjacent keys. But in reality, there might be some sort of distribution over the adjacent keys.
# ----------------------------------------------------------------------------------------------------------------------
# Negation and contracted forms of negation exempted from transformation.
NEGATION_WORDS = {
    # Negation
    "not", "no", "never", "neither", "nobody", "none", "nor", "nothing", "nowhere", "cannot",
    # Contractions
    "n't",
    # Cases when ' is dropped in contractions
    "aint", "cant", "dont", "doesnt", "didnt", "wont",
    "isnt", "arent", "wasnt", "werent",
    "hasnt", "havent", "hadnt", "couldnt",
    "shouldnt", "wouldnt"
}
# Since we are using word_tokenize, "didn't" will be tokenized into "did" and "n't". In this case, "did" should be
# exempted from transformation, because otherwise we could have "ddin't" or "dadn't" after detokenization. But the
# meaning also changes. So, we need to exempt "did" from transformation only when it is immediately followed by "n't".
# So, we need a helper set, which contain words that could be contracted with "n't" to form a negation.
NEGATION_WORDS_HELPER = {
    "do", "does", "did", "is", "are", "ai", "was", "were", "have", "has", "had",
    "ca", "could", "will", "wo", "would", "sha", "should", "may", "might", "must"
}
# ----------------------------------------------------------------------------------------------------------------------
# Words that people drop in speech and writing in casual settings or for faster speed.
DROPPABLE_WORDS = {
    "the", "a", "an", "i", "it", "we", "you", "that", "this", "is", "was", "are", "were"
}
# ======================================================================================================================


# ======================================================================================================================
# Helper functions
# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
# Synonyms Helper
def get_synonyms(word, rng=RNG):
    """
    Get synonyms from wordnet. But only sample and return one.
    :param rng:
    :param word:
    :return:
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name()
            if name.lower() != word.lower() and name.isalpha():
                synonyms.add(name)
    if synonyms:
        sampled = str(rng.choice(sorted(synonyms))) # sorted instead of list() cus list() pick different with same seed
        # Since we are using cased tokenizer in main.py, preserving case is needed.
        if word.isupper():
            return sampled.upper()
        elif word.istitle():
            return sampled.capitalize()
        return sampled.lower()
    return word  # return og if nothing found.
# ----------------------------------------------------------------------------------------------------------------------
# Typo Helper
def get_typo(word, rng=RNG):
    """
    Applies one of three with uniform prob:
      1. Swapping adjacent letters (e.g., 'the' -> 'hte' or "teh")
      2. Hitting adjacent keys on QWERTY keyboard
      3. Double-hit (e.g., "that" -> "thaat")
    NOTE:
    Real-world human behaviors are really nuanced. Deterministically perturbing the dataset this way makes our data
    out-of-distribution, but it doesn't make it more in-distribution with actual human behavior either. A lot of my
    design choices are just based on my own habits and my assumptions and intuitions. So, the transformed dataset
    will be out-of-distribution for the original IMDB dataset, but I cannot say that it will be more in-distribution
    with actual human behavior either. One pretty clear red flag is the probability assigned to drop/typo/synonym.
    These are numbers that I picked based on pure "feeling".
    :param rng:
    :param word:
    :return:
    """
    chars = list(word)
    is_title = word.istitle()
    typo_type = rng.choice([1, 2, 3]) if len(chars) >=2 else rng.choice([2, 3])
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Type 1
    if typo_type == 1:
        i = rng.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        # So, when we type on a phone, the first letter is automatically capitalized.
        # Or, when we type on a computer, it is highly unlikely that we type first letter without holding the shift but
        # type the second while holding. So, we will preserve this: "The" -> "Hte" instead of "hTe".
        if is_title and i == 0:
            chars[0], chars[1] = chars[0].upper(), chars[1].lower()
        return ''.join(chars)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Type 2
    elif typo_type == 2:
        i = rng.randint(0, len(chars) - 1)
        c = chars[i].lower()
        if c in KEYBOARD_ADJACENT:
            replacement = rng.choice(KEYBOARD_ADJACENT[c])
            chars[i] = replacement.upper() if chars[i].isupper() else replacement  # also preserve case.
            return "".join(chars)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Type 3
    i = rng.randint(0, len(chars) - 1)
    chars.insert(i, chars[i])
    return "".join(chars)
# ======================================================================================================================


# ======================================================================================================================
# Custom transformation function
# ======================================================================================================================
def custom_transform(example, rng=RNG):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in PDF
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    text = example["text"]
    words = word_tokenize(text)
    transformed = []
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for i, word in enumerate(words):
        word_lower = word.lower()
        # -------------------------------------------------------
        # Contraction and negation
        next_lower = words[i + 1].casefold() if i + 1 < len(words) else ""
        contraction_helper = (word_lower in NEGATION_WORDS_HELPER and next_lower == "n't") or (word_lower == "can" and next_lower == "not")
        # Have to add a special case for cannot, since it is word_tokenize into "can" and "not".
        # Negation, punctuation, numbers
        if (not word.isalpha()) or (word_lower in NEGATION_WORDS) or contraction_helper:
            transformed.append(word)
            continue
        # -------------------------------------------------------
        # Droppable words
        if word_lower in DROPPABLE_WORDS:
            random_value = rng.random()
            if random_value < 0.05:
                continue
            elif random_value < 0.20:
                transformed.append(get_typo(word, rng))
            else:
                transformed.append(word)
        # -------------------------------------------------------
        # All other words
        else:
            random_value = rng.random()
            if random_value < 0.15:
                transformed.append(get_typo(word, rng))
            elif random_value < 0.30:
                transformed.append(get_synonyms(word, rng))
            else:
                transformed.append(word)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    example["text"] = TreebankWordDetokenizer().detokenize(transformed)

    ##### YOUR CODE ENDS HERE ######

    return example
# ======================================================================================================================

# if __name__ == "__main__":
    # pass
    # ==================================================================
    # Test word_tokenize and TreebankWordDetokenizer
    # text = "It didn't happen. It wasnt true! This cannot be true. It isn't good."
    # tokens = word_tokenize(text)
    # print(tokens)
    # og_text = TreebankWordDetokenizer().detokenize(tokens)
    # print(og_text)
    # ==================================================================
    # Test Bert tokenizer (cased since main.py)
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    # text = [
    #     # "It didn't happen. It wasnt true! well-chosen",
    #     # "It didn't happen . It wasnt true!",
    #     # "This", "this", "THIS"
    #     "This isn't good."
    # ]
    # for t in text:
    #     tokens = tokenizer.tokenize(t)
    #     ids = tokenizer.encode(t, add_special_tokens=False)
    #     print(tokens)
    #     print(ids)
    #     print(tokenizer.convert_tokens_to_string(tokens))
    # So, it doesn't matter if there is a space before the punctuation.
    # Therefore, no steps to address this is needed for the word_tokenize
    # and and TreebankWordDetokenizer.
    # So, for synonym multiple words synonym, remove them?
    #
    # ==================================================================
    # Test get_synonyms
    # words = ["good", "bad", "Happy", "Film", "awful"]
    # for w in words:
    #     print(w, get_synonyms(w))
    # iterations = 20
    # for _ in range(iterations):
    #     print(get_synonyms(words[3]))
    # So, some words have synonyms that are multiple words. Or words with hyphens.
    # Remove them from being sampled as replacement just for simplicity.
    # ==================================================================