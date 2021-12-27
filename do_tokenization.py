import argparse
import logging
import json
import os
import pickle
import re
from tqdm import tqdm, trange

from KoBERTNER.tokenization_kobert import KoBertTokenizer


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def html_remover(paragraph):
    html_pattern = re.compile("<[0-9a-zA-Z!@#$%^&*()\\/ \"=]+>")
    founded = list(set(html_pattern.findall(paragraph)))
    for word in founded:
        paragraph = paragraph.replace(word," ")
    unique = ['《','》','(',')']
    for word in unique:
        paragraph = paragraph.replace(word," ")
    return paragraph


class SQuADExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 qas_id,
                 question_text,
                 question_entities_strset,
                 doc_tokens,
                 passage_entities,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.passage_entities = passage_entities
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.question_entities_strset = question_entities_strset

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        return s


def read_squad_examples(input_file, is_training):
    """"Read a SQuAD json file into a list of SQuADExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
        
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    
    examples = []
    for entry in tqdm(input_data):
        paragraph_raw = entry["context"]
        temp_raw = entry["context"]
        paragraph_text = html_remover(paragraph_raw)
        line_split = paragraph_text.split("\n")
        line_list = []
        for line in line_split:
            line = line.strip()
            if line == " " or line == "" or line == "  " or line == "   ":
                continue
            else:
                line_list.append(line)
        paragraph_text = "\n".join(line_list)
        doc_tokens = []
        doc_location = []
        char_to_word_offset = []
        raw_to_char = dict()
        prev_is_whitespace = True
        for char in paragraph_text:
            if is_whitespace(char):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    if len(doc_tokens) != 0:
                        start = temp_raw.index(doc_tokens[-1])
                        end = temp_raw.index(doc_tokens[-1])+len(doc_tokens[-1])
                        doc_location.append([doc_tokens[-1],start,end])
                        temp_str = "".join(["ㅱ" for x in range(len(doc_tokens[-1]))])
                        temp_raw = temp_raw.replace(doc_tokens[-1],temp_str,1)
                        raw_to_char[(start,end)] = len(doc_tokens) - 1
                    doc_tokens.append(char)
                else:
                    doc_tokens[-1] += char
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        last_start = temp_raw.index(doc_tokens[-1])
        last_end = temp_raw.index(doc_tokens[-1])+len(doc_tokens[-1])
        doc_location.append([doc_tokens[-1],last_start,last_end])
        temp_str = "".join(["ㅱ" for x in range(len(doc_tokens[-1]))])
        temp_raw = temp_raw.replace(doc_tokens[-1],temp_str,1)
        passage_entities = []
        for entity in entry['context_entities']:
            entity_start_offset = entity['start']
            entity_end_offset = entity['end']
            entity_text = entity['text']
            assert entity_text == paragraph_raw[entity_start_offset: entity_end_offset]
            entity_text_split = entity_text.split(" ")
            start_str = entity_text_split[0]
            end_str = entity_text_split[-1]
            start_str_pos = (entity_start_offset,entity_start_offset+len(start_str))
            end_str_pos = (entity_end_offset - len(end_str), entity_end_offset)
            if start_str_pos in raw_to_char.keys() and end_str_pos in raw_to_char.keys():
                passage_entities.append({'orig_text': entity_text, 
                                    'start_position': raw_to_char[start_str_pos], 
                                    'end_position': raw_to_char[end_str_pos]})
                
        for qa in entry["qas"]:
            qas_id = qa["id"]
            question_text = qa["question"]
            question_entities_strset = set([entity_info["text"] for entity_info in qa["question_entities"]])
            start_position = None
            end_position = None
            orig_answer_text = None
            if is_training:
                answer = qa["answer"]
                orig_answer_text = answer["text"]
                answer_offset = answer["answer_start"]
                answer_length = len(orig_answer_text)
                answer_text_split = orig_answer_text.split(" ")
                start_positon = (answer_offset, answer_offset + len(answer_text_split[0]) )
                end_position = (answer_offset + answer_length - len(answer_text_split[-1]),answer_offset + answer_length)
                # Only add answers where the text can be exactly recovered from the
                # document. If this CAN'T happen it's likely due to weird Unicode
                # stuff so we will just skip the example.
                #
                # Note that this means for training mode, every example is NOT
                # guaranteed to be preserved.
            example = SQuADExample(
                qas_id=qas_id,
                question_text=question_text,
                question_entities_strset=question_entities_strset,
                doc_tokens=doc_tokens,
                passage_entities=passage_entities,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position)
            examples.append(example)
    return examples


def _improve_entity_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_entity_text):
    """Returns token-level tokenized entity spans that better match the annotated entity."""
    tok_entity_text = " ".join(tokenizer.tokenize(orig_entity_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_entity_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _is_real_subspan(start, end, other_start, other_end):
    return (start >= other_start and end < other_end) or (start > other_start and end <= other_end)



def match_query_entities(query_tokens, entities_tokens):
    # transform query_tokens list into a whitespace separated string
    query_string = " ".join(query_tokens)
    offset_to_tid_map = []
    tid = 0
    for char in query_string:
        offset_to_tid_map.append(tid)
        if char == ' ':
            tid += 1

    # transform entity_tokens into whitespace separated strings
    entity_strings = set()
    for entity_tokens in entities_tokens:
        entity_strings.add(" ".join(entity_tokens))
    
    # do matching
    results = []
    for entity_string in entity_strings:
        start = 0
        while True:
            pos = query_string.find(entity_string, start)
            if pos == -1:
                break
            token_start, token_end = offset_to_tid_map[pos], offset_to_tid_map[pos] + entity_string.count(' ')
            # assure the match is not partial match (eg. "ville" matches to "danville")
            if " ".join(query_tokens[token_start: token_end + 1]) == entity_string:
                results.append((token_start, token_end))
            start = pos + len(entity_string)
            if len(entity_string) == 0:
                break
    
    # filter out a result span if it's a subspan of another span
    no_subspan_results = []
    for result in results:
        if not any([_is_real_subspan(result[0], result[1], other_result[0], other_result[1]) for other_result in results]):
            no_subspan_results.append((" ".join(query_tokens[result[0]: result[1] + 1]), result[0], result[1]))
    assert len(no_subspan_results) == len(set(no_subspan_results))

    return no_subspan_results


# the further tokenization process when generating features
def tokenization_on_examples(examples, tokenizer):

    tokenization_result = []
    for example in tqdm(examples):
        # do tokenization on raw question text
        query_subtokens = []
        query_sub_to_ori_index = [] # mapping from sub-token index to token index
        query_tokens = tokenizer.tokenize(example.question_text)
        for index, token in enumerate(query_tokens):
            for sub_token in tokenizer.tokenize(token):
                query_subtokens.append(sub_token)
                query_sub_to_ori_index.append(index)
        
        # do tokenization on whitespace tokenized document
        document_tokens = []
        document_subtokens = []
        document_sub_to_ori_index = []
        document_up_to_ori_index = [] # map unpunc token index to tokenized token index
        for unpunc_tokenized_tokens in example.doc_tokens:
            tokens = tokenizer.tokenize(unpunc_tokenized_tokens) # do punctuation tokenization
            document_up_to_ori_index.append(len(document_tokens))
            for token in tokens:
                for sub_token in tokenizer.tokenize(token):
                    document_subtokens.append(sub_token)
                    document_sub_to_ori_index.append(len(document_tokens))
                document_tokens.append(token)

        # generate token-level document entity index
        document_entities = []
        for entity in example.passage_entities:
            entity_start_position = document_up_to_ori_index[entity['start_position']]
            entity_end_position = None
            if entity['end_position'] < len(example.doc_tokens) - 1:
                entity_end_position = document_up_to_ori_index[entity['end_position'] + 1] - 1
            else:
                entity_end_position = len(document_tokens) - 1
            (entity_start_position, entity_end_position) = _improve_entity_span(
                document_tokens, entity_start_position, entity_end_position, tokenizer, entity['orig_text'])
            document_entities.append((entity['orig_text'], entity_start_position, entity_end_position)) # ('Trump', 10, 10)

        # match query entities (including tagged and document entities)
        entities_tokens = []
        for question_entity_str in example.question_entities_strset:
            entities_tokens.append(tokenizer.tokenize(question_entity_str))
        for document_entity in document_entities:
            entities_tokens.append(document_tokens[document_entity[1]: document_entity[2] + 1])
        query_entities = match_query_entities(query_tokens, entities_tokens) # [('trump', 10, 10)]

        tokenization_result.append({
            'id': example.qas_id,
            'query_tokens': query_tokens,
            'query_subtokens': query_subtokens,
            'query_entities': query_entities,
            'query_sub_to_ori_index': query_sub_to_ori_index,
            'document_tokens': document_tokens,
            'document_subtokens': document_subtokens,
            'document_entities': document_entities,
            'document_sub_to_ori_index': document_sub_to_ori_index,
        })
    
    return tokenization_result
    


parser = argparse.ArgumentParser()

parser.add_argument("--output_dir", default='tokens', type=str, 
                    help="The output directory to dump tokenization results.")
parser.add_argument("--train_file", default='./data/kor/korquad2.1_train_00.tagged.json', type=str, help="SQuAD json for training. E.g., train-v1.1.json")
parser.add_argument("--predict_file", default='./data/kor/korquad2.1_train_01.tagged.json', type=str,
                    help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
# parser.add_argument("--do_lower_case", default=False, action='store_true',
#                     help="Whether to lower case the input text. Should be True for uncased "
#                          "models and False for cased models.")
# parser.add_argument('--dump_token', action='store_true', help='whether dump the token-level tokenization result')
# parser.add_argument('--dump_subtoken', action='store_true', help='whether dump the subtoken-level tokenization result, with its mapping with token-level result')
args = parser.parse_args([])
# make output directory if not exist
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

# We do both cased and uncased tokenization
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
for do_lower_case in (True, False):
    train_examples = read_squad_examples(input_file=args.train_file, is_training=True)
    train_tokenization_result = tokenization_on_examples(
        examples=train_examples,
        tokenizer=tokenizer)
    with open(os.path.join(args.output_dir, 'train.tokenization.{}.data'.format('uncased' if do_lower_case else 'cased')), 'wb') as fout:
        pickle.dump(train_tokenization_result, fout)

    eval_examples = read_squad_examples(input_file=args.predict_file, is_training=False)
    eval_tokenization_result = tokenization_on_examples(
        examples=eval_examples,
        tokenizer=tokenizer) 
    with open(os.path.join(args.output_dir, 'dev.tokenization.{}.data'.format('uncased' if do_lower_case else 'cased')), 'wb') as fout:
        pickle.dump(eval_tokenization_result, fout)

