import functools
import math

import numpy as np
import tqdm

FEEDBACK_STRING = """Enter the result:
\"x\" for gray, \"y\" for yellow and \"g\" for green.
Write \"i\" if I chose an invalid word.
Write \"c\" to list all candidates.
Write \"r\" to restart.
Write \"o\" to guess another word.\n"""


def get_word_frequencies():
    with open("./word_frequencies.csv") as f:
        words = f.readlines()
    words = [x[:-1] for x in words]
    words = words[1:]
    words = [x.split(",") for x in words]
    words = [x for x in words if len(x[0]) == 5]
    words = [(x[0], x[1]) for x in words]
    totals = sum(int(x[1]) for x in words)
    return {x[0]: int(x[1]) / totals for x in words}


def initialise_candidates():
    with open("./vocab.txt") as f:
        words = f.readlines()
    words = [x[:-1].lower() for x in words]
    words = [x for x in words if len(x) == 5]
    words = [x for x in words if x.islower()]
    words = list(set(words))
    frequencies = get_word_frequencies()
    words = {x: frequencies.get(x, 0.0) for x in words}
    words = {x: y for x, y in words.items() if y != 0.0}
    return normalise(words)


def normalise(posterior):
    totals = sum(val for _, val in posterior.items())
    return {x[0]: x[1] / totals for x in posterior.items()}


def score_word(word, posterior, num_samples=32):
    total_posterior_score = 0
    total_green_score = 0
    actual_samples = min(num_samples, len(posterior))
    words = list(posterior)
    probs = [posterior[x] for x in words]
    for sampled_target in np.random.choice(words, size=actual_samples, p=probs):
        result = get_result(word, sampled_target)
        simulated_posterior = refine_posterior(word, result, posterior)
        run_score = sum(val for _, val in simulated_posterior.items())
        if run_score == 0:
            run_score = sum(val for _, val in simulated_posterior.items())
        total_posterior_score += run_score
        total_green_score += result.count("g")
    return len(posterior) / total_posterior_score, total_green_score / len(posterior)


def proposal(prior, posterior, posterior_clip=128):
    if "tares" in posterior:
        return "tares", None
    max_prob = max(val for _, val in posterior.items())
    if (max_prob > 0.8) or (len(posterior) == 2):
        return [x for x in posterior if posterior[x] == max_prob][0], None
    if len(posterior) > posterior_clip:
        words = list(posterior)
        probs = [posterior[x] for x in words]
        sampled_words = np.random.choice(words, size=posterior_clip, p=probs)
        posterior_sample = {x: posterior[x] for x in sampled_words}
        posterior_sample = normalise(posterior_sample)
    else:
        posterior_sample = posterior
    scores = {word: score_word(word, posterior_sample) for word in tqdm.tqdm(prior)}
    max_score = max(x[1] for x in scores.items())
    return [x for x in scores if scores[x] == max_score][0], scores


def get_result(guess, actual):
    counts = {x: actual.count(x) for x in actual}
    output = ["x" for _ in range(5)]
    for i, (guess_letter, actual_letter) in enumerate(zip(guess, actual)):
        if guess_letter == actual_letter:
            output[i] = "g"
            counts[guess_letter] -= 1
    for i, guess_letter in enumerate(guess):
        if output[i] == "x" and counts.get(guess_letter, 0) > 0:
            output[i] = "y"
            counts[guess_letter] -= 1
    return "".join(output)


def refine_posterior(guess, result, posterior):
    candidates = list(posterior)
    present_word = [x for i, x in enumerate(guess) if result[i] != "x"]
    present_guess_counts = {x: present_word.count(x) for x in present_word}
    for i, color in enumerate(result):
        letter = guess[i]
        if color == "x":
            candidates = [
                x for x in candidates if x.count(letter) <= present_guess_counts.get(letter, 0)
            ]
        elif color == "y":
            candidates = [
                x
                for x in candidates
                if x.count(letter) >= present_guess_counts[letter] and x[i] != letter
            ]
        elif color == "g":
            candidates = [x for x in candidates if x[i] == letter]
    return {x: posterior[x] for x in candidates}


def prompt_for_feedback():
    return input(FEEDBACK_STRING)


def play_wordle():
    prior = initialise_candidates()
    posterior = initialise_candidates()
    posterior = normalise(posterior)
    while True:
        guess, scores = proposal(prior, posterior)
        print(f'I guess "{guess}".\n')
        feedback = prompt_for_feedback()
        while feedback == "i":
            scores[guess] = -math.inf, -math.inf
            max_score = max(x[1] for x in scores.items())
            guess = [x for x in scores if scores[x] == max_score][0]
            print(f'I guess "{guess}".\n')
            feedback = prompt_for_feedback()
        while feedback == "c":
            print(",".join(posterior))
            feedback = prompt_for_feedback()
        if feedback == "o":
            guess = input("What word did you guess?")
            feedback = input(
                """ Enter the result: \"x\" for gray, \"y\" for yellow and \"g\" for green.\n"""
            )
            posterior = refine_posterior(guess, feedback, posterior)
            posterior = normalise(posterior)
        if feedback == "r":
            prior = initialise_candidates()
            posterior = initialise_candidates()
            posterior = normalise(posterior)
        else:
            posterior = refine_posterior(guess, feedback, posterior)
            posterior = normalise(posterior)


if __name__ == "__main__":
    play_wordle()
