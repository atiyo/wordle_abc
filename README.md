# Approximate Bayesian Computation Wordle Bot

A Wordle bot that plays using principles inspired by Approximate Bayesian Computation.

Essentially the bot chooses a word which is expected to maximally 'narrow down' a given
prior/posterior density of words with each guess until a point where it is left with a single word of
suitably high posterior density.

To use, call `python wordle_abc.py`. You'll also need to include a dictionary of new-line delimited
valid words in a file called `vocab.txt`, and a CSV of word frequencies (from which prior
distributions of words are derived) in `word_frequencies.csv` (first column word, second column
frequency, with a header included).

Copies of `vocab.txt` or `word_frequencies.txt` are not provided out of respect for underlying
licenses of sources from which I derived them for my personal use.

A decent dictionary is pretty important. Using `/usr/share/dict/words` didn't cut the mustard in my
testing.
