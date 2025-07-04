import pandas as pd
from datasets import load_dataset
from sklearn.utils import resample

def load_data(num_samples=None):
    """
    Input: num_samples (int) - number of training samples. (uniform sampling)
    Returns: train_df (pd.DataFrame), test_df (pd.DataFrame)
    """
    dataset = load_dataset('glue', 'sst2')

    train_data = dataset['train']
    test_data = dataset['validation']

    train_df = pd.DataFrame(train_data)[['sentence', 'label']]

    if num_samples:
        num_classes = train_df['label'].nunique()
        samples_per_class = num_samples // num_classes

        sampled_train_df = pd.DataFrame()

        for class_label in train_df['label'].unique():
            class_data = train_df[train_df['label'] == class_label]
            sampled_class_data = resample(class_data,
                                          replace=False,
                                          n_samples=samples_per_class,
                                          random_state=42)
            sampled_train_df = pd.concat([sampled_train_df, sampled_class_data])

        train_df = sampled_train_df
    test_df = pd.DataFrame(test_data)
    
    return train_df, test_df
