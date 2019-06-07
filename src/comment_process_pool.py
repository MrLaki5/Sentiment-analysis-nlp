import threading
import pandas as pd
import tokenizer
from stemmer import stemmer
import json
import logging

logging.basicConfig(level=logging.DEBUG)


# Function for getting original and stemmed tokens of one comment
def zip_comment(raw_text, temp_file_name):
    # Generate tokens for data set
    tokens_original = tokenizer.text_to_tokens(raw_text)
    token_text = ""
    for i in tokens_original:
        token_text = token_text + i + "\n"
    token_text = stemmer.call_stemmer(token_text, temp_file_name)
    tokens_stemmed = token_text.split("\n")
    cnt = 0
    while cnt < len(tokens_original):
        if tokens_stemmed[cnt] == "":
            del tokens_stemmed[cnt]
            del tokens_original[cnt]
        else:
            cnt += 1

    return tokens_stemmed, tokens_original


# Class for managing all process threads
class CommentProcessPool:

    def __init__(self, thread_num):
        self.thread_num = thread_num
        self.process_pool = []

    # Start process threads to calculate specific data set
    def start(self, data_set):
        padding = int(len(data_set) / self.thread_num)
        start_index = 0
        end_index = padding
        cnt = 0
        while cnt < self.thread_num:
            self.process_pool.append(ProcessWorker(data_set, start_index, end_index, "Thread" + str(cnt)))
            cnt += 1
            start_index = end_index
            end_index += padding
            if cnt is (self.thread_num - 1):
                end_index = len(data_set)
        cnt = 0
        while cnt < self.thread_num:
            self.process_pool[cnt].start()
            cnt += 1

    # Interrupt processing of threads
    def interrupt(self):
        cnt = 0
        while cnt < len(self.process_pool):
            try:
                with self.process_pool[cnt].lock:
                    self.process_pool[cnt].interrupted = True
                self.process_pool[cnt].join()
            except Exception as ex:
                pass
            cnt += 1

    # Wait and get processed results
    def get_data(self):
        cnt = 0
        return_data = []
        while cnt < len(self.process_pool):
            try:
                self.process_pool[cnt].join()
            except Exception as ex:
                pass
            return_data += self.process_pool[cnt].results
            cnt += 1
        return return_data


# Thread for processing comments and converting them to tokens
class ProcessWorker(threading.Thread):

    def __init__(self, data, start_index, end_index, name=""):
        super(ProcessWorker, self).__init__()
        self.data = data
        self.start_index = start_index
        self.end_index = end_index
        self.interrupted = False
        self.lock = threading.Lock()
        self.results = []
        self.name = name

    def run(self):
        data_cnt = self.start_index
        while True:
            if data_cnt >= self.end_index:
                break
            with self.lock:
                if self.interrupted:
                    break
            raw_text = self.data.loc[data_cnt, 'Text']
            sentiment_class = self.data.loc[data_cnt, 'class-att']
            tokens_stemmed, tokens_original = zip_comment(raw_text, self.name)
            logging.debug("ProcessWorker, name: " + self.name + ", processed comment: " +
                          str(data_cnt - self.start_index) + ", out of: " + str(self.end_index - self.start_index))
            self.results.append(
                {
                    "tokens_original": tokens_original,
                    "tokens_stemmed": tokens_stemmed,
                    "class_att": sentiment_class
                }
            )
            data_cnt += 1
        logging.info("ProcessWorker: thread exiting, name: " + self.name)
