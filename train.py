from pprint import pprint
from transformers import AutoTokenizer
import datasets
import pandas as pd

# dataset_path = "/content/drive/MyDrive/A2/ibm_data"



def read_conll(file):
    examples = []
    # example = {col: [] for col in INPUT_COLUMNS}
    idx = 0
    example = {"id":idx, "tokens": [], "ner_tags":[]}
    
    with open(file) as f:
        for line in f:
            if line.startswith("-DOCSTART-") or line == "\n" or not line:
                assert len(example["tokens"]) == len(example["ner_tags"])
                examples.append(example)
                idx+=1
                example = {"id":idx, "tokens": [], "ner_tags":[]}
            else:
                row_cols = line.split()
                assert len(row_cols) == 3
                example["tokens"].append(row_cols[0])
                example["ner_tags"].append(row_cols[-1])

    return examples

def get_dataset(dataset_path):
    train_data = read_conll( dataset_path+"/test_500_v2.conll")
    test_data = read_conll( dataset_path+"/train_1500_v2.conll")
    ner_feature = datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-TASK",
                                "I-TASK",
                                "B-METRIC",
                                "I-METRIC",
                                "B-DATASET",
                                "I-DATASET"
                            ]
                        )
                    )

    token_feature = datasets.Sequence(datasets.Value("string"))
    id_feature = datasets.Value("string")
    train_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=train_data), features=datasets.Features({
        "id":id_feature,
        "ner_tags":ner_feature,
        "tokens" : token_feature
    }))
    test_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=test_data), features=datasets.Features({
        "id":id_feature,
        "ner_tags":ner_feature,
        "tokens" : token_feature
    }))
    return train_dataset, test_dataset








if __name__ == "__main__":
    dataset_path = "/Users/afreenshaikh/Library/CloudStorage/GoogleDrive-afreens@andrew.cmu.edu/.shortcut-targets-by-id/1tZMZ1hVZ12FuHdPiu88Shf3zG1ZgrK47/A2/ibm_data"
    train_dataset, test_dataset = get_dataset(dataset_path)
    task = "ner"
    label_list = train_dataset.features[f"{task}_tags"].feature.names



    model_checkpoint = "distilbert-base-uncased"
    batch_size = 16
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    import transformers
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    example = train_dataset[4]
    tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
    print(tokens)

    print(len(example[f"{task}_tags"]), len(tokenized_input["input_ids"]))
    print(tokenized_input.word_ids())





