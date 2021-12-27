import os
import json
import argparse
import logging
import urllib
import sys
from tqdm import tqdm
import re

sys.path.append("./KoBERTNER/")
from KoBERTNER import predict_sentence

INPUT_FILE = "./data/kor/korquad2.1_train_00.json"
OUTPUT_FILE = "./data/kor"
MODEL_DIR = './model'

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument("--input_file", default="sample_pred_in.txt", type=str, help="Input file for prediction")
parser.add_argument("--output_file", default="sample_pred_out.txt", type=str, help="Output file for prediction")
parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
pred_config = parser.parse_args(["--input_file",INPUT_FILE,"--output_file",OUTPUT_FILE,"--model_dir",MODEL_DIR])

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default='output', type=str, 
                        help="The output directory to store tagging results.")
    parser.add_argument("--train_file", default='../../data/SQuAD/train-v1.1.json', type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default='../../data/SQuAD/dev-v1.1.json', type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    return parser.parse_args()

# transform corenlp tagging output into entity list
# some questions begins with whitespaces and they are striped by corenlp, thus begin offset should be added.
# sent Structure -> (Word, NER)
def parse_output(sentence, tagging_output, raw_text): 
    entities = []
    for sent in tagging_output['NERS']:
        state = 'O'
        if sent[1] != 'O':
            sentence_pos = raw_text.index(sentence)
            try:
                start_pos = sentence_pos + sentence.index(sent[0])
                end_pos = start_pos + len(sent[0])
            except:
                start_pos = sentence_pos + sentence.index(sent[0][1:-1])
                end_pos = start_pos + len(sent[0][1:-1])
            entities.append({'text' : sent[0], 'start' : start_pos, 'end' : end_pos})
    return entities


def html_remover(paragraph):
    html_pattern = re.compile("<[0-9a-zA-Z!@#$%^&*()\\/ ]+>")
    founded = list(set(html_pattern.findall(paragraph)))
    for word in founded:
        paragraph = paragraph.replace(word," ")
    unique = ['《','》','(',')']
    for word in unique:
        paragraph = paragraph.replace(word," ")
    return paragraph


def tagging(dataset):
    predictor = predict_sentence.NERPredictor(pred_config)
    for article in tqdm(dataset['data']):
        raw_text = article['context']
        context_entities_list = []
        for context in raw_text.split("\n"):
            sentence = html_remover(context)
            if sentence == "":
                continue
            context_tagging_output = predictor.predict(sentence)
            # assert the context length is not changed
            context_entities = parse_output(context, context_tagging_output, raw_text)
            if len(context_entities) != 0:
                context_entities_list.extend(context_entities)
        article['context_entities'] = context_entities_list
        for qa in article['qas']:
            question = qa['question']
            question_tagging_output = predictor.predict(question)
            question_entities = parse_output(question, question_tagging_output, question)            
            qa['question_entities'] = question_entities


ftrain = open('./data/kor/korquad2.1_train_00.json', 'r', encoding='utf-8')
trainset = json.load(ftrain)
fdev = open('./data/kor/korquad2.1_train_01.json', 'r', encoding='utf-8')
devset = json.load(fdev)


for dataset, path, name in zip((trainset, devset), ('data/kor/korquad2.1_train_00.json','data/kor/korquad2.1_train_01.json'), ('train', 'dev')):
    tagging(dataset)
    output_path = os.path.join(OUTPUT_FILE, "{}.tagged.json".format(os.path.basename(path)[:-5]))
    json.dump(dataset, open(output_path, 'w', encoding='utf-8'))

