import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

def main():
    model_dir = "./model/7b-chat"

    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    model = LlamaForCausalLM.from_pretrained(model_dir)

    prompt = """"
    Translate to Korean: Hello, I'm student.

    korean: 안녕하세요, 저는 학생입니다.

    Translate to Korean: A large language model (LLM) is a 
    language model notable for its ability to achieve general-purpose 
    language generation and other natural language processing tasks such 
    as classification

    korean: 
    """

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(**inputs, max_length=256, num_return_sequences=1, num_beams=5)

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded_output)

if __name__ == "__main__":
    main()
