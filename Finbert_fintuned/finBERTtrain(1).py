import os
import torch
from transformers import BertForSequenceClassification
from finbert.finbert import Config, FinBert
import logging

logging.basicConfig(level=logging.ERROR)

logging.getLogger("finbert.utils").setLevel(logging.ERROR)
logging.getLogger("finbert.finbert").setLevel(logging.ERROR)

config = Config(
    data_dir="train",
    bert_model=BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3),
    model_dir="trained_model",
    max_seq_length=64,
    train_batch_size=16,
    eval_batch_size=16,
    learning_rate=5e-5,
    num_train_epochs=30,
    warm_up_proportion=0.1,
    no_cuda=False,
    do_lower_case=True,
    seed=42,
    gradient_accumulation_steps=1,
    output_mode='classification',
    discriminate=True,
    gradual_unfreeze=True,
    encoder_no=12,
    base_model='bert-base-uncased'
)

finbert = FinBert(config)

label_list = [0, 1, 2]
finbert.prepare_model(label_list)

train_examples = finbert.get_data('train')
validation_examples = finbert.get_data('validation')
print(f"Number of training examples: {len(train_examples)}")
print(f"Number of validation examples: {len(validation_examples)}")
print(train_examples)

model = finbert.create_the_model()

finbert.train(train_examples, model)

output_model_file = os.path.join(config.model_dir, "finetuned_model.bin")
torch.save(model.state_dict(), output_model_file)

print(f"微调完成，模型已保存到 {output_model_file}")