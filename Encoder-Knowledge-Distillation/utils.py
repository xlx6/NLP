from transformers import (
    DebertaV2PreTrainedModel,
    DebertaV2Config
    )
import re
from torch import nn
from datasets import load_dataset
from teacher_train import DebertaScratch
from transformers.modeling_outputs import SequenceClassifierOutput

def get_data(tokenizer, path):
    if path.lower() == 'sst2':
        target = 'sentence'
        dataset = load_dataset('glue', 'sst2')
        train_dataset, test_dataset = dataset['train'], dataset['validation']
    elif path.lower() == 'imdb':
        target = 'text'
        dataset = load_dataset('imdb')
        train_dataset, test_dataset = dataset['train'], dataset['test']
    else:
        raise ValueError('Dataset not supported')

    tag_re = re.compile(r'<.*?>')

    def preprocess_function(examples):
        return tokenizer(
            examples[target], 
            truncation=True,
            max_length=512)

    def clean_and_tokenize_batch(examples):
        cleaned = [tag_re.sub('', text) for text in examples[target]]
        return tokenizer(
            cleaned,
            truncation=True,
            max_length=512
        )

    tokenized_train = train_dataset.map(
        clean_and_tokenize_batch if target == 'text' else preprocess_function, 
        batched=True
    )
    tokenized_test = test_dataset.map(
        clean_and_tokenize_batch, 
        batched=True
    )

    return tokenized_train, tokenized_test

def create_student(teacher_model, use_teacher_weights=True): 
    configuration = teacher_model.config.to_dict() 
    
    configuration["num_hidden_layers"] //= 2 
    
    configuration = DebertaV2Config.from_dict(configuration) 
    
    student_model = type(teacher_model)(configuration) 
    if use_teacher_weights:
        copy_deberta_weights(teacher_model, student_model) 
    
    return student_model 

def copy_deberta_weights(teacher, student): 
    student.deberta.embeddings.load_state_dict(teacher.deberta.embeddings.state_dict())
    teacher_layers = teacher.deberta.encoder.layer
    student_layers = student.deberta.encoder.layer
    for i, student_layer in enumerate(student_layers):
        teacher_layer = teacher_layers[i*2]
        student_layer.load_state_dict(teacher_layer.state_dict())
    if hasattr(student.deberta, 'pooler') and hasattr(teacher.deberta, 'pooler'):
        student.deberta.pooler.load_state_dict(teacher.deberta.pooler.state_dict())

    student.classifier.load_state_dict(teacher.classifier.state_dict())
    student.dropout.load_state_dict(teacher.dropout.state_dict())

if __name__ == '__main__':
    teacher = DebertaScratch.from_pretrained('./model_checkpoints_sst2/checkpoint-500')
    student = create_student(teacher, use_teacher_weights=False)
    student.save_pretrained('./student_basemodel_sst2_zero')
    
    