from openai import OpenAI
import argparse
import time

def main():

    FLAGS = parser.parse_args()

    client = OpenAI(base_url="http://localhost:3000/v1", api_key="None")

    model_name = "llama3"
    messages = [
        {
            "role": "user",
            "content": FLAGS.query,
        },
    ]

    chat_completion = client.chat.completions.create(
        model=model_name,
        stream=FLAGS.stream,
        messages=messages,
        max_tokens=256,
    )

    if FLAGS.stream:
        response = ""
        for event in chat_completion:
            content = event.choices[0].delta.content
            if content:
                print(content)
                response += content
                time.sleep(0.05)
    else:
        # print(chat_completion.choices[0])
        response = chat_completion.choices[0].message.content

    print(f"---->Full response:\n\n{response}")

    print("PASS")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        required=False,
        default="Fix grammar error in the following sentence:\n\nYou're a very famous comediant that tells great store to entertain people.",
        help="Query text",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        required=False,
        default=False,
        help="Stream output.",
    )

    main()
