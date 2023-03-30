import numpy as np
import openai
openai.api_key = "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"

import urllib.request
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np

#kobert
from kobert.utils import get_tokenizer

#transformers
from transformers import AdamW # ???

#GPU 사용
device = torch.device("cuda:0")

import pickle

# 저장된 Vocabulary 불러오기
with open('/Users/leesangyup/Desktop/project_3/emotion_recog/text_emotion_models/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

#Setting parameters
max_len = 64
batch_size = 64

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

#kobert 학습모델 만들기
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))[1]
        #bert model returns 'last_hidden_state' and 'pooler_output'

        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


# load the saved model from disk
with open('/Users/leesangyup/Desktop/project_3/emotion_recog/text_emotion_models/model.pkl', 'rb') as f:
    model = pickle.load(f)

#새로운 문장 테스트
#토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)








# 영어/한글로 번역해주는 함수
def Transword(inputtext, lan1, lan2):
    client_id = "drdBnYic4NLWtEbuPnVi" # 개발자센터에서 발급받은 Client ID 값
    client_secret = "iudbsGsLIv" # 개발자센터에서 발급받은 Client Secret 값
    encText = urllib.parse.quote(inputtext)
    data = f"source={lan1}&target={lan2}&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        result = response_body.decode('utf-8')
        d = json.loads(result)
        translate_result = d['message']['result']['translatedText']
        # print(translate_result)
        return(translate_result)
    else:
        return("Error Code:" + rescode)


# 일기 요약해주는 함수
def summarize_text(text):
    prompt = (f"Please summarize the following text in one sentence:\n{text}")
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=prompt,
      max_tokens=30,
      n=1,
      stop=None,
      temperature=0.7,
    )
    summary = response.choices[0].text.strip()
    return summary



# 감정을 영어로 추출해주는 함수
def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model.eval()
    with torch.no_grad():

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)

            valid_length= valid_length
            label = label.long().to(device)

            out = model(token_ids, valid_length, segment_ids)


            test_eval=[]
            emotion_english = []
            for i in out:
                logits=i
                logits = logits.detach().cpu().numpy()

                if np.argmax(logits) == 0:
                    test_eval.append("불안이")
                    emotion_english.append('anxiety')
                elif np.argmax(logits) == 1:
                    test_eval.append("당황이")
                    emotion_english.append('panic')
                elif np.argmax(logits) == 2:
                    test_eval.append("분노가")
                    emotion_english.append('anger')
                elif np.argmax(logits) == 3:
                    test_eval.append("슬픔이")
                    emotion_english.append('sadness')
                elif np.argmax(logits) == 5:
                    test_eval.append("행복이")
                    emotion_english.append('happy')

            emotion_result = "당신이 쓴 일기 내용에서 " + test_eval[0] + " 느껴집니다." 
            my_emotion = emotion_english[0]
            return emotion_result, my_emotion



# 요약된 일기와 감정을 받아서 덕담 한 마디 던져주는 함수
def give_word_for_me(summurized_diary, my_emotion):
    # prompt = (f"{summurized_diary}. This is the diary I wrote today. and my emotion is {my_emotion}. Please analyze the diary I wrote today and the feelings I delivered and say a word of praise, comfort, criticism, empathy or advice to me today based on it.")
    prompt = (f"[{summurized_diary}]. This is the diary I wrote today. Please say something to me based on the contents of my diary. It doesn't matter whether it's comfort, advice, empathy, etc.")

    response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=30,
    n=1,
    stop=None,
    temperature=0.7,
    )
    word_for_me = response.choices[0].text.strip()
    return word_for_me



# 일기 입력하기
dairy = input("일기를 입력해주세요 : ")

# 일기를 영어로 번역
translate_result = Transword(dairy, 'ko', 'en')
# 한 문장으로 요약
summurized_diary = summarize_text(translate_result)

#일기 감정 분석 결과
emotion_result = predict(dairy)
# 일기에서 감정을 영어로추출
my_emotion = emotion_result[1]
# 감정 결과를 보여줌
emotion_print = emotion_result[0]
# 요약과 감정을 받아 덕담 한 마디(영어 -> 한글 번역)
word_for_me = give_word_for_me(summurized_diary, my_emotion)
word_for_me_final = Transword(word_for_me, 'en', 'ko')

print('')
print('<일기 내용 감정분석 결과>')
print(emotion_print)
print('')
print('-당신을 위한 한 마디-')
print(word_for_me_final)

