from openai import OpenAI
import torch


def main():
    client = OpenAI(base_url="http://localhost:3000/v1", api_key="None")

    model_name = "text-embedding"
    input_texts = [
        "Fix grammar error in the following sentence",
        "Fix grammar error in the following sentence",
        "You are a very famous comedian that tells great stories to entertain people",
        "You are a comedian that enjoys telling stories to cheer up people",
    ]

    results = client.embeddings.create(
        model=model_name,
        input=input_texts,
    )

    # print(results)

    embed_matrix = torch.Tensor([item.embedding for item in results.data])

    # one and two are identical
    score_1 = torch.cosine_similarity(embed_matrix[0], embed_matrix[1], dim=0)

    score_2 = torch.cosine_similarity(embed_matrix[0], embed_matrix[2], dim=0)
    score_3 = torch.cosine_similarity(embed_matrix[1], embed_matrix[3], dim=0)

    # last pair are similar
    score_4 = torch.cosine_similarity(embed_matrix[2], embed_matrix[3], dim=0)

    print(score_1)
    print(score_2)
    print(score_3)
    print(score_4)

    assert round(score_1.item(), 4) == 1.0
    assert round(score_2.item(), 4) < 0.3
    assert round(score_3.item(), 4) < 0.3
    assert round(score_4.item(), 4) >= 0.7

    print("PASS")

if __name__ == "__main__":
    main()
