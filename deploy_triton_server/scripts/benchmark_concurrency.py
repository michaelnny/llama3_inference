import requests
import time
# from multiprocessing import Pool
import concurrent.futures
import numpy as np
import random
import json
import argparse
import csv
import os

# Simple hack to support running without installing as a package, so we can import the tokenizer
from pathlib import Path
import sys
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from src.tokenizer import Tokenizer


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






def make_request(args):
    url, stream, max_gen_len, tokenizer_path, print_output = args

    tokenizer = Tokenizer(tokenizer_path)
    prompt = random.choice(queries)

    start_time = time.time()

    response = requests.post(
        url=url,
        data=json.dumps({
                "max_tokens": max(128, max_gen_len),
                "stop_words": ["<|eot_id|>"],
                "top_p": 0.7,
                "stream": stream,
                "text_input": chat_template.format(prompt),
            }),
        headers={
            "Content-Type": "application/json",
        }
    )

    total_time = time.time() - start_time

    response_text = response.content.decode('utf-8')
    # print(response_text)

    if response_text.startswith("data: "):
        # Strip the 'data: ' prefix
        response_text = response_text[len("data: "):].strip()

    response_json = json.loads(response_text)

    answer = response_json['text_output']

    out_ids = tokenizer.encode(answer, allowed_special='all')
    num_out_tokens = len(out_ids)

    if print_output:
        print('-'*60)
        print(f'INPUT TEXT: {prompt}')
        print(f'OUTPUT TEXT: {answer}')
        print('\n')

    return total_time, num_out_tokens



def run_tests(args, concurrency):
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        requests = [(args.url, args.stream, args.max_gen_len, args.tokenizer_path, args.print_output) for _ in range(concurrency)]

        latencies = []
        num_tokens = []
        futures = [executor.submit(make_request, req) for req in requests]

        for future in concurrent.futures.as_completed(futures):
            latency, token_count = future.result()
            latencies.append(latency)
            num_tokens.append(token_count)

    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p70 = np.percentile(latencies, 70)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)

    token_rate = np.sum(num_tokens) / np.sum(latencies)

    return {
        'concurrency': concurrency,
        'max_gen_len': args.max_gen_len,
        'token_rate': token_rate,
        'p50_latency': p50,
        'p70_latency': p70,
        'p90_latency': p90,
        'p99_latency': p99
    }



def main(args):

    # Define your concurrency range
    concurrency_range = range(2, args.max_concurrency+1, 2)  # range from 2 to N, with step size 2

    # Ensure the directory exists
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.output_csv, 'w', newline='') as csvfile:
        # header columns
        fieldnames = ['concurrency', 'max_gen_len', 'token_rate', 'p50_latency', 'p70_latency', 'p90_latency', 'p99_latency']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()

        for concurrency in concurrency_range:
            result = run_tests(args, concurrency)
            csv_writer.writerow(result)
            csvfile.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", type=str, default='http://localhost:8000/v2/models/llama3_ensemble/generate_stream', help="Triton server URL"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default='./src/tokenizer.model', help=""
    )
    parser.add_argument(
        "--max_concurrency", type=int, default=32, help="Maximum number of concurrent requests"
    )

    parser.add_argument(
        "--stream", action='store_true', default=False, help="Enable or disable streaming"
    )

    parser.add_argument(
        "--print_output", action='store_true', default=False, help="Enable or disable print generated content"
    )
    parser.add_argument(
        "--output_csv", type=str, default='./logs/concurrency_performance_results.csv'
    )

    parser.add_argument(
        "--max_gen_len", type=int, default=128, help="Maximum generation token length"
    )
   
    args = parser.parse_args()

    main(args)
