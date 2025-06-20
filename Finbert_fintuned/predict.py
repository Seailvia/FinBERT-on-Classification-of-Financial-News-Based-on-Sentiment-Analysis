# from finbert.finbert import predict
# from transformers import AutoModelForSequenceClassification
# import argparse
# import os


# parser = argparse.ArgumentParser(description='Sentiment analyzer')

# parser.add_argument('-a', action="store_true", default=False)

# parser.add_argument('--text_path', type=str, help='Path to the text file.')
# parser.add_argument('--output_dir', type=str, help='Where to write the results')
# parser.add_argument('--model_path', type=str, help='Path to classifier model')

# # args = parser.parse_args()

# # if not os.path.exists(args.output_dir):
# #     os.mkdir(args.output_dir)


# # with open(args.text_path,'r', encoding='utf-8', errors='ignore') as f:
# #     text = f.read()

# # model = AutoModelForSequenceClassification.from_pretrained(args.model_path,num_labels=3,cache_dir=None)

# # output = "predictions.csv"
# # predict(text,model,write_to_csv=True,path=os.path.join(args.output_dir,output))

# parser.add_argument('--text_path', type=str, help='Path to the text file.')
# parser.add_argument('--output_dir', type=str, default='output', help='Where to write the results')  # 设置默认值
# parser.add_argument('--model_path', type=str, help='Path to classifier model')

# args = parser.parse_args()

# if not os.path.exists(args.output_dir):
#     os.mkdir(args.output_dir)

# with open(args.text_path, 'r', encoding='utf-8', errors='ignore') as f:
#     text = f.read()

# model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=3, cache_dir=None)

# output = "predictions.csv"
# predict(text, model, write_to_csv=True, path=os.path.join(args.output_dir, output))

from finbert.finbert import predict
from transformers import AutoModelForSequenceClassification
import argparse
import os

parser = argparse.ArgumentParser(description='Sentiment analyzer')

parser.add_argument('--text_path', type=str, help='Path to the text file.')
parser.add_argument('--output_dir', type=str, default='output', help='Where to write the results')  # 设置默认值
parser.add_argument('--model_path', type=str, help='Path to classifier model')

args = parser.parse_args()

if args.text_path is None:
    raise ValueError("The --text_path argument is required. Please provide the path to the text file.")

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

with open(args.text_path, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=3, cache_dir=None)

output = "predictions.csv"
predict(text, model, write_to_csv=True, path=os.path.join(args.output_dir, output))