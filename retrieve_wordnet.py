import pickle
import argparse
import os
import logging
import string
from tqdm import tqdm

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


offset_to_wn18name_dict = {} 
fin = open('wordnet-mlj12-definitions.txt')
for line in fin:
    info = line.strip().split('\t')
    offset_str, synset_name = info[0], info[1]
    offset_to_wn18name_dict[offset_str] = synset_name


def KorWordNet(offset_to_wn18name_dict):
    fin = open('kwn_synset_list.txt',encoding = 'utf8')
    kor_offset = dict()
    exist_check = dict()
    for idx, line in enumerate(fin):
        if idx == 0:
            continue
        info = line.strip().split("\t")
        info_key = info[0][:-2]
        for word in info[3].split(", "):
            if word not in kor_offset.keys():
                kor_offset[word] = [info_key]
            else:
                kor_offset[word].append(info_key)
#         if info_key in offset_to_wn18name_dict.keys():
#             content = info[3].split(",")
#             new_content = []
#             for word in content:
#                 new = word+"_"+info[1]
#                 if new not in exist_check.keys():
#                     exist_check[new] = 1
#                 else:
#                     exist_check[new] += 1
#                 new_content.append(word+"_"+info[1]+"_"+str(exist_check[new]))
#             kor_offset[info_key] = new_content
    return kor_offset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_token', type=str, default='./tokens/train.tokenization.uncased.data', help='token file of train set')
    parser.add_argument('--kor_wordnet',type=str,default='./kwn_synset_list.txt',help='korean wordnet files which developed in KAIST')
    parser.add_argument('--eval_token', type=str, default='./tokens/dev.tokenization.uncased.data', help='token file of dev set')
    parser.add_argument('--output_dir', type=str, default='output_record/', help='output directory')
    parser.add_argument('--no_stopwords', action='store_true', help='ignore stopwords')
    parser.add_argument('--ignore_length', type=int, default=0, help='ignore words with length <= ignore_length')
    args = parser.parse_args([])

    # initialize mapping from offset id to wn18 synset name
    offset_to_wn18name_dict = {} 
    fin = open('wordnet-mlj12-definitions.txt')
    for line in fin:
        info = line.strip().split('\t')
        offset_str, synset_name = info[0], info[1]
        offset_to_wn18name_dict[offset_str] = synset_name
    logger.info('Finished loading wn18 definition file.')

    kor_offset = KorWordNet(offset_to_wn18name_dict)
    # load pickled samples
    logger.info('Begin to load tokenization results...')
    train_samples = pickle.load(open(args.train_token, 'rb'))
    dev_samples = pickle.load(open(args.eval_token, 'rb'))
    logger.info('Finished loading tokenization results.')
    
    # build token set
    all_token_set = set()
    for sample in train_samples + dev_samples:
        for token in sample['query_tokens'] + sample['document_tokens']:
            all_token_set.add(token)
    logger.info('Finished making tokenization results into token set.')

    
    # retrive synsets
    logger.info('Begin to retrieve synsets...')
    token2synset = dict()
    stopword_cnt = 0
    punctuation_cnt = 0
    for token in tqdm(all_token_set):
        if token in set(string.punctuation):
            logger.info('{} is punctuation, skipped!'.format(token))
            punctuation_cnt += 1
            continue        
        if args.ignore_length > 0 and len(token) <= args.ignore_length:
            logger.info('{} is too short, skipped!'.format(token))
            continue
        ###### 여기서부터 바꿔야 함. 한글 토큰 -> synset의 오프셋으로 바꾼다-> 그 오프셋들을 통하여서 A단어 : 명사1 명사2 이렇게
        #### 현재는 그냥 OFFSET을 저장하도록 조치함. 추후 수정이 필요할 지도 모름.
        try:
            synsets = kor_offset[token]
        except:
            continue
#         wn18synset_names = []
#         for synset in synsets:
#             if synset in offset_to_wn18name_dict:
#                 wn18synset_names.append(offset_to_wn18name_dict[offset_str])
        if len(synsets) > 0:
            token2synset[token] = synsets
    logger.info('Finished retrieving sysnets.')
    logger.info('{} / {} tokens retrieved at lease 1 synset. {} stopwords and {} punctuations skipped.'.format(len(token2synset), len(all_token_set), stopword_cnt, punctuation_cnt))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'retrived_synsets.data'), 'wb') as fout:
        pickle.dump(token2synset, fout)    
    logger.info('Finished dumping retrieved synsets.')


if __name__ == '__main__':
    main()
