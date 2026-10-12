
import time

import runpod
from transformers import pipeline

MODEL = "Horizon-Labs/multilingual-zeroshot-small"

print("Loading classification model...")
classifier = pipeline(
    "zero-shot-classification",
    model=MODEL,
    device="cpu",
)
print("Model ready!")


def handler(job):
    data = job["input"]
    message = data["message"]
    tools = data["tools"]

    if not tools:
        return {"error": "No tools provided"}

    descriptions = [tool["description"] for tool in tools]

    start = time.perf_counter()

    result = classifier(
        message,
        candidate_labels=descriptions,
        hypothesis_template="The user's request means: {}",
        multi_label=False,
    )

    elapsed_ms = round((time.perf_counter() - start) * 1000)

    selected_description = result["labels"][0]
    selected_index = descriptions.index(selected_description)

    return {
        "tool": tools[selected_index]["name"],
        "score": round(result["scores"][0], 4),
        "inference_ms": elapsed_ms,
    }


runpod.serverless.start({"handler": handler})
