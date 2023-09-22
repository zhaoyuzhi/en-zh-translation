import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# save a list to a txt
def text_save(content, filename, mode = 'a'):
    # try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

# read a txt expect EOF
def text_readlines(filename):
    # try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, encoding="gb18030", errors="ignore")
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content

def translate_single_sentence(inp_text, tokenizer, model, device):
    encode_text = tokenizer(inp_text, max_length=512, truncation=True, padding=True, return_tensors='pt').to(device)
    out = model.generate(**encode_text, max_length=1024)
    out_text = tokenizer.decode(out[0])
    out_text = out_text.replace("<pad> ", "").replace("<pad>", "")
    return out_text

if __name__  == '__main__':

    # the input file paths
    file_path = 'caption/llava_describe_advertisement_image.txt'
    save_path = file_path.split('.txt')[0] + '_translated.txt'
    model_path = 'Helsinki-NLP/opus-mt-en-zh'

    # build tokenizer and model
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    # translation
    src_text = text_readlines(file_path)
    translated_text = []
    for i in range(len(src_text)):
        out = translate_single_sentence(src_text[i], tokenizer, model, device)
        print(i, len(src_text), out)
        translated_text.append(out)
    text_save(translated_text, save_path)