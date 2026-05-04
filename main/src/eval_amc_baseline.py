from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, re

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

dataset = load_dataset("AI-MO/aimo-validation-amc", split="train")
problems = list(dataset)[:100]
print(f"Loaded {len(problems)} AMC problems\n")

def extract_answer(text):
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else None

results = []
correct = 0

for i, problem in enumerate(problems):
    question = problem["problem"]
    gold_answer = str(problem["answer"]).strip()
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, temperature=1.0, do_sample=False)
    response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    predicted = extract_answer(response)
    is_correct = predicted == gold_answer
    if is_correct:
        correct += 1
    results.append({"problem_id": i, "question": question, "gold_answer": gold_answer, "model_response": response, "predicted_answer": predicted, "correct": is_correct})
    print(f"[{i+1}/100] {'✓' if is_correct else '✗'}  Gold: {gold_answer}  |  Predicted: {predicted}")

model_short = MODEL_NAME.replace("/", "_")
out_file = f"results_{model_short}.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nAccuracy: {correct}/100 = {correct}%")
print(f"Saved to: {out_file}")
