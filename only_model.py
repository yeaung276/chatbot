from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
import torch

tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

while True:
    user_input = input("You: ")

    tokens = tokenizer(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**tokens)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response) 