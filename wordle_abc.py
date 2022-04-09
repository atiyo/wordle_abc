import math

import numpy as np
import tqdm
import multiprocessing as mp

FEEDBACK_STRING = """Enter the result:
\"x\" for gray, \"y\" for yellow and \"g\" for green.
Write \"i\" if I chose an invalid word.
Write \"c\" to list all candidates.
Write \"r\" to restart.
Write \"o\" to guess another word.\n"""


def read_words(filepath):
    with open(filepath) as f:
        words = f.readlines()
    words = [x[:-1].lower() for x in words]
    words = [x for x in words if len(x) == 5]
    words = [x for x in words if x.islower()]
    return set(words)


def initialise_posterior():
    return read_words("./posterior.txt")


def initialise_prior():
    posterior = read_words("./posterior.txt")
    non_posterior = read_words("./non_posterior_vocab.txt")
    return posterior.union(non_posterior)


def score_word(word, posterior):
    total_posterior_score = 0
    total_green_score = 0
    for sampled_target in posterior:
        result = get_result(word, sampled_target)
        simulated_posterior = refine_posterior(word, result, posterior)
        run_score = -len(simulated_posterior)
        if run_score == 0:
            run_score = -len(posterior)
        total_posterior_score += run_score
        total_green_score += result.count("g")
    return total_posterior_score, total_green_score / len(posterior)


def make_global_posterior(posterior):
    global global_posterior
    global_posterior = posterior


def par_func(word):
    return word, score_word(word, global_posterior)


def proposal(prior, posterior, guess_num, feedback):
    if guess_num == 1:
        return "roate", None
    if guess_num == 2:
        if feedback == "xxxxx":
            return "slimy", None
        if feedback == "yxxxx":
            return "sculk", None
        if feedback == "xyxxx":
            return "snool", None
        if feedback == "xxyxx":
            return "lysin", None
        if feedback == "xxxyx":
            return "shunt", None
        if feedback == "xxxxy":
            return "silen", None
        if feedback == "gxxxx":
            return "rugby", None
        if feedback == "xgxxx":
            return "bludy", None
        if feedback == "xxgxx":
            return "slick", None
        if feedback == "xxxgx":
            return "hinds", None
        if feedback == "xxxxg":
            return "sling", None
    with mp.Pool(initializer=make_global_posterior, initargs=(posterior,)) as pool:
        output = list(tqdm.tqdm(pool.imap_unordered(par_func, prior), total=len(prior)))
    scores = {x[0]: x[1] for x in output}
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
    return set(candidates)


def prompt_for_feedback():
    return input(FEEDBACK_STRING)


def play_wordle():
    prior = initialise_prior()
    posterior = initialise_posterior()
    guess_num = 1
    feedback = 'xxxxx'
    while True:
        guess, scores = proposal(prior, posterior, guess_num, feedback)
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
            guess_num += 1
        if feedback == "r":
            prior = initialise_prior()
            posterior = initialise_posterior()
            guess_num = 1
        else:
            posterior = refine_posterior(guess, feedback, posterior)
            guess_num += 1


if __name__ == "__main__":
    play_wordle()
