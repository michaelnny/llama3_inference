import requests
import time
from multiprocessing import Pool
import random
import json
import argparse


queries = [
    "What is the score of the last football match between Chelsea and Arsenal?",
    "List 5 features of a good team leader.",
    "Identify 5 different shapes.",
    "Summarize the importance of monitoring air quality",
    "Given a list of items, sort them in descending order of their cost.",
    "Answer the following question: How can machine learning be used in marketing?",
    "Suggest a way that I can make my car more fuel efficient.",
    "Check for spelling and grammar errors in the sentence",
    "Come up with a list of kitchen utensils.",
    "Provide two examples of how to express regrets in a formal letter.",
    'Generate at least 15 synonyms of the word "planned".',
    "Summarize the following text in four to six sentences.",
    "Write a slogan that describes the importance of embracing diversity.",
    "Rewrite this sentence so that it has a more dramatic emphasis: 'She was walking along the street'.",
    "Classify this news headline.",
    "Summarize the following article about climate change in three sentences:",
    "Create an output array which is an element wise multiplication of two given arrays.",
    "What is the most expensive item in the given list?",
    "Select the correct word to fill in the blank: The air pollution in the city is getting ____.",
    "Construct a plan for something you'd like to accomplish.",
    "List the advantages of using cryptocurrency.",
    "Imagine you are playing a board game. Tell me what kind of game it is.",
    "Compose three sentences about the following topic:",
    "Sum up the following article in one sentence.",
    "Compose an apology letter to an old friend.",
    "Design a classification algorithm to detect a message as spam or ham.",
    "Name three limitations of the given topic.",
    'Model a complex sentence, with "although" and a noun clause.',
    "Name the type of food that is usually served in a bento box.",
    "Describe the four stages of natural selection.",
    "Generate a sentence about a fictional character.",
    "Name 3 endangered species native to Australia",
    "Based on the given data, what is the probable cause of the accident?",
    "Create a short dialog between a teacher and a student discussing a missed assignment.",
    "Generate a new idea for a product based on the following inputs.",
    "Arrange a series of steps to explain how to cook a microwave recipe.",
    "Given an imperative sentence, change it to an interrogative sentence",
    "Create a list of adjectives to describe the given noun.",
    "Explain the concept of net neutrality.",
    "Generate a list of steps for starting a new business.",
    "Generate a script for a radio ad.",
    "Reverse a string without using any built-in functions.",
    "Determine the probability that a team will win the game given the equation P(x) = (x + 1) / (x + 3).",
    "Write five questions related to the given article.",
    "Given two pieces of text, select one of two types of transition words (addition or comparison) in order to connect them.",
    "Create a list of five things that a student should avoid in order to succeed in academia.",
    "Name 5 ways to conserve water.",
    "Convert this hexadecimal value to its equivalent decimal number.",
    "Summarize the text by outputting one sentence.",
    "Give a convincing argument for.",
    "Create a tagline that summarizes the idea behind a shopping website.",
    "Create a set of steps for how to make homemade ice cream.",
    "Provide example code for the following data structure: stack.",
    "Describe the overall feeling of the work environment at a tech startup",
    "Describe the problems of human trafficking in the given paragraph.",
    "Explain the process for reversing a linked-list in Java.",
    "Paraphrase the following sentence without changing the meaning: He was foolish enough to cross the street without looking.",
    "Turn this sentence into a question: The man was wearing a hat.",
    "Rewrite the following sentence using proper English grammar and spelling.",
    "Create a product design for a throw pillow.",
    "Categorize the following as either an input or output device.",
    "Research and cite 3 examples of medicinal plants commonly used in traditional medicine.",
    "Define what botnet is in one sentence.",
    "Create a search query for finding a product.",
    "Identify four positive impacts that artificial intelligence can have on the healthcare industry",
    "Please rate this service on a scale of 1 to 5.",
]


chat_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{0}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def make_request(stream):
    start_time = time.time()

    response = requests.post(
        url="http://localhost:8000/v2/models/ensemble/generate_stream",
        data=json.dumps(
            {
                "max_tokens": 256,
                "stop_words": ["<|eot_id|>"],
                "top_p": 0.7,
                "stream": stream,
                "text_input": chat_template.format(random.choice(queries)),
            }
        ),
    )

    latency = time.time() - start_time

    for line in response.iter_lines():
        print(line)

    return latency

def main(args):
    with Pool(processes=args.num_concurrency) as pool:
        requests = [args.stream for _ in range(args.num_concurrency)]

        latencies = []
        for latency in pool.imap(make_request, requests):
            latencies.append(latency)

    print(f"Average latency: {sum(latencies) / len(latencies):.4f} seconds")
    print(f"Min latency: {min(latencies):.4f} seconds")
    print(f"Max latency: {max(latencies):.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_concurrency", type=int, default=8, help="Number of concurrent requests"
    )

    parser.add_argument(
        "--stream", action='store_true', default=False, help="Enable or disable streaming"
    )
   
    args = parser.parse_args()

    main(args)
