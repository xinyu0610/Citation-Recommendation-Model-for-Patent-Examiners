import os


FILE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

KGS = {
    'HowNet': os.path.join(FILE_DIR_PATH, 'kgs/HowNet.spo'),
    'CnDbpedia': os.path.join(FILE_DIR_PATH, 'kgs/CnDbpedia.spo'),
    'Medical': os.path.join(FILE_DIR_PATH, 'kgs/Medical.spo'),
    'Patent': os.path.join(FILE_DIR_PATH, 'kgs/patent_info.spo'),
    'Title': os.path.join(FILE_DIR_PATH, 'kgs/titles.spo'),
    'Assignee': os.path.join(FILE_DIR_PATH, 'kgs/assignees.spo'),
    'Classification': os.path.join(FILE_DIR_PATH, 'kgs/classifications.spo'),
    'Date': os.path.join(FILE_DIR_PATH, 'kgs/dates.spo'),
    'Inventors': os.path.join(FILE_DIR_PATH, 'kgs/inventors.spo'),
}

MAX_ENTITIES = 1

# Special token words.
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'
ENT_TOKEN = '[ENT]'
SUB_TOKEN = '[SUB]'
PRE_TOKEN = '[PRE]'
OBJ_TOKEN = '[OBJ]'

NEVER_SPLIT_TAG = [
    PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN,
    ENT_TOKEN, SUB_TOKEN, PRE_TOKEN, OBJ_TOKEN
]
