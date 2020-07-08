import requests
#from lxml.html import fromstring
import bs4
import re

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, ids, input_ids, input_mask, segment_ids, label_id):
        self.ids = ids
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def find_urls(text):
    url_finder = re.compile(r'https?://\S+|www\.\S+')
    urls = url_finder.findall(text)
    if len(urls)==0: return ''
    else: return urls

def get_title_from_url(url):
    try: r=requests.get(url, allow_redirects=True, timeout=1)
    except: return ''
    if r.status_code == 200:
        title = bs4.BeautifulSoup(r.text, "html5lib").title
        if title is not None:
            return re.sub('<[/]*title[^>]*>', '', str(title))
        else: return ''
    else: return ''

def convert_example_to_feature(example_row, typ='train'):
    # return example_row
    example, label_map, max_seq_length, tokenizer, output_mode = example_row
    urls = find_urls(example.text_a)
    if type(urls) == list:
        for url in urls:
            title = get_title_from_url(url)
            try:text = example.text_a.replace(url, title)
            except: print(url)
    else: text = example.text_a
    # remove hashtag symbol(#,@)
    tokens_a=[tok for tok in tokenizer.tokenize(re.sub(r'(^|\s)#[a-zA-Z0-9]+|(^|\s)@[a-zA-Z0-9]+',' ',text.strip()))]
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2: tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if output_mode == "classification":
        label_id = label_map.get(example.label, 0)
    else:
        raise KeyError(output_mode)

    return InputFeatures(
        ids=example.guid,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)