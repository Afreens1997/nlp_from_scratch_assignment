import spacy
from spacy.matcher import Matcher
import uuid


nlp_evaluation_metrics = [
    "Precision",
    "Recall",
    "F1",
    "Accuracy",
    "True Positive Rate","TPR",
    "True Negative Rate","TNR",
    "False Positive Rate", "FPR",
    "False Negative Rate","FNR",
    "Precision-Recall Curve",
    "ROC Curve",
    "AUC-ROC",
    "AUC-PR",
    "Confusion Matrix",
    "MCC",
    "Matthews Correlation Coefficient",
    "Cohen's Kappa",
    "Average Precision","AP",
    "Hamming Loss",
    "Multi-class Log Loss","Cross-Entropy",
    "Mean Squared Error", "MSE",
    "RMSE", "Root Mean Squared Error",
    "BLEU","Bilingual Evaluation Understudy",
    "WER" ,"Word Error Rate",
    "METEOR",
    "ROUGE", "Recall-Oriented Understudy for Gisting Evaluation",
    "CIDEr", "Consensus-based Image Description Evaluation",
    "BLEURT",
    "TER", "Translation Edit Rate",
    "Translation Recall",
    "Translation Precision",
    "Gleitman and Gleitman's M Score",
    "(M^2)",
    "Word Embedding Similarity",
    "Cluster Evaluation Metrics",
    "Perplexity",
    "Edit Distance",
    "Levenshtein Distance",
    "Edit Distance with Transpositions",
    "Damerau-Levenshtein Distance",
    "ASR-WER",
    "Word Error Rate for ASR",
    "Translation Fluency",
    "Translation Adequacy",
    "METEOR Paraphrase Metric",
    "METEOR",
    "SARI",
    "System-level Automatic Reviewer",
    "Dependency Parse Tree Metrics",
    "Semantic Textual Similarity Metrics",
    "WMD",
    "Word Mover's Distance",
    "Coverage Metrics",
    "ROUGE-L",
    "Sequence Accuracy",
    "Cosine Similarity",
    "MRR",
    "Mean Reciprocal Rank",
    "Information Retrieval Metrics",
    "Response Coherence",
    "Bilingual Evaluation Understudy",
    "BEER",
    "LSA Similarity",
    "Latent Semantic Analysis",
    "LSA",
    "Lexical Diversity Metrics",
    "NIST",
    "ROUGE-W",
    "Bilingual Lexicon Induction Metrics",
    "Cross-Lingual Word Embedding Evaluation Metrics",
    "Document Embedding Similarity",
    "Dependency Parsing Evaluation Metrics",
    "Topic Coherence Metrics",
    "Word Sense Disambiguation Metrics",
    "Language Model Evaluation Metrics",
    "Coreference Resolution Metrics",
    "Named Entity Linking (NEL) Metrics",
    "Text Classification Metrics",
    "Information Extraction Metrics",
    "DCG",
    "Discounted Cumulative Gain",
    "Precision at K",
    "Average Precision",
    "MUC",
    "B-Cubed",
    "CEAF",
    "UAS",
    "LAS",
    "BLI",
    "MAP",
    "TTR",
    "Type-Token Ratio",
    "Semantic Textual Similarity",
    "STS",
    "Unlabeled Attachment Score",
    "UAS",
    "Labeled Attachment Score",
    "LAS",
    "Adjusted Rand Index",
    "ARI",
    "Normalized Mutual Information",
    "NMI",
    "accuracy",
    "precision",
    "recall",
    "F1",
    "ROC",
    "AUC",
    "MSE",
    "MAE",
    "MRR",
    "DCG",
    "NDCG",
    "Correlation",
    "PSNR",
    "SSIM",
    "IoU",
    "Perplexity",
    "BLEU",
    "Logarithmic Loss",
    "Area Under Curve",
    "True positive rate",
    "True Negative Rate",
    "False positive Rate",
    "F1",
    "Confusion Matrix",
    "Spearman correlations",
    "macro f1",
    "micro f1",
    "Mean Absolute Error",
    "Mean Squared Error",
    "Root Mean Square Error",
    "RMSE",
    "Root Mean Square Logarithmic Error",
    "R2-Score",
    "ROUGE"
]

nlp_evaluation_tasks = [
    "Named Entity Recognition",
    "NER",
    "POS",
    "Part-of-Speech Tagging",
    "Sentiment Analysis",
    "Machine Translation",
    "MT",
    "Speech Recognition",
    "ASR",
    "Text Classification",
    "Text Summarization",
    "Coreference Resolution",
    "Dependency Parsing",
    "Question Answering",
    "QA",
    "Language Modeling",
    "Entity Linking",
    "NEL",
    "Information Retrieval",
    "IR",
    "Text Generation",
    "Named Entity Linking",
    "NEL",
    "Aspect-Based Sentiment Analysis",
    "ABSA",
    "Text Similarity",
    "Document Classification",
    "Coreference Resolution",
    "Semantic Role Labeling",
    "SRL",
    "Text Segmentation",
    "Dialogue Systems",
    "Language Understanding",
    "LU",
    "Natural Language Understanding",
    "NLU",
    "NLG",
    "Natural Language Generation",
    "Dialog Policy Learning",
    "POL",
    "Text Simplification",
    "Keyphrase Extraction",
    "Paraphrase Generation",
    "Grammatical Error Correction",
    "GEC",
    "Machine Comprehension",
    "MC",
    "Text Clustering",
    "Emotion Analysis",
    "Event Extraction",
    "Multilingual Translation",
    "Relation Extraction",
    "Speaker Identification",
    "Topic Modeling",
    "Word Sense Disambiguation",
    "WSD",
    "Speech Synthesis",
    "TTS",
    "Cross-Lingual Information Retrieval",
    "CLIR",
    "Conversational Agents"
]


hyperparameters = [
    "Learning rate",
    "α",
    "Number of epochs",
    "n",
    "Batch size",
    "m",
    "Momentum",
    "Weight decay",
    "L2 regularization",
    "Dropout rate",
    "Dropout",
    "Number of layers",
    "L",
    "Number of units",
    "neurons per layer",
    "Activation function",
    "Optimizer",
    "Loss function",
    "Learning rate schedule",
    "Kernel size",
    "Number of filters",
    "Strides",
    "Padding",
    "Initialization method",
    "Batch normalization",
    "Gradient clipping threshold",
    "Embedding dimension",
    "Sequence length",
    "Vocabulary size",
    "Maximum tree depth",
    "Minimum samples per leaf",
    "Number of clusters",
    "Number of estimators",
    "Number of neighbors",
    "k",
    "C",
    "Regularization strength",
    "Gamma",
    "Kernel coefficient",
    "Number of topics",
    "Latent dimensions",
    "Min support",
    "Minimum lift threshold",
    "Number of principal components",
    "PCA",
    "Number of latent factors",
    "Window size",
    "Alpha",
    "Learning rate",
    "Subsample ratio",
    "Depth of decision trees",
    "Minimum child weight",
    "Learning rate",
    "Number of iterations",
    "Subsample",
    "Number of trees",
    "Subsample for feature selection",
    "Minimum impurity decrease",
    "Number of layers",
    "Attention heads",
    "Hidden dimension",
    "Feedforward dimension",
    "Number of self-attention layers",
    "Number of encoder layers",
    "Number of decoder layers"
]

nlp_datasets = [
    "MNIST",
    "CIFAR-10",
    "CIFAR-100",
    "ImageNet",
    "UCI Machine Learning Repository",
    "PASCAL VOC",
    "COCO",
    "Common Objects in Context",
    "IMDB Movie Reviews",
    "Reuters",
    "New York Times Annotated Corpus",
    "Wikipedia",
    "Gutenberg Project",
    "20 Newsgroups",
    "Amazon Customer Reviews",
    "Yelp Reviews",
    "Stanford Sentiment Treebank",
    "SQuAD",
    "Stanford Question Answering Dataset",
    "GloVe",
    "Word2Vec",
    "FastText",
    "GPT-2",
    "BERT",
    "Eurlex",
    "DBpedia",
    "TREC",
    "SNLI",
    "Stanford Natural Language Inference",
    "MultiNLI",
    "AG News",
    "DBPedia",
    "ImageNet",
    "Flickr8k and Flickr30k",
    "OpenSubtitles",
    "WikiText",
    "Penn Treebank",
    "BooksCorpus",
    "BNC",
    "British National Corpus",
    "Brown Corpus",
    "LFW",
    "Labeled Faces in the Wild",
    "CelebA",
    "Oxford Pets",
    "VGGFace",
    "Street View House Numbers",
    "SVHN",
    "Cityscapes",
    "MIT Scene Parsing Benchmark",
    "Fashion MNIST",
    "OpenAI GPT-3 Datasets",
    "Kaggle Datasets",
    "Google AI Datasets",
    "Facebook AI Datasets",
    "CoNLL",
    "Twitter Sentiment Analysis",
    "Twitter Emotion Analysis",
    "Reddit Comments",
    "Quora Question Pairs",
    "Amazon Product Reviews",
    "Yelp Polarity Reviews",
    "IMDb Movie Reviews",
    "SpamAssassin",
    "TREC Question Classification",
    "Reuters News Dataset",
    "AG News",
    "Yelp Dataset",
    "SNLI",
    "Stanford Natural Language Inference",
    "MultiNLI",
    "SWAG",
    "Situation With Adversarial Generations",
    "Penn Treebank",
    "Brown Corpus",
    "Cornell Movie Dialogs Corpus",
    "Quora Insincere Questions Classification",
    "OpenSubtitles",
    "20 Newsgroups",
    "Linguistic Data Consortium",
    "LDC",
    "Conversational Intelligence Challenge",
    "ConvAI",
    "ParaNMT Corpus",
    "Parallel Neural Machine Translation",
    "CoQA",
    "Conversational Question Answering",
    "NLI Benchmark",
    "Natural Language Inference",
    "SNIPS Spoken Language Understanding",
    "SCARE",
    "Synthetic Corpus for Abstractive and REsummarization",
    "ConLL-2003 Named Entity Recognition",
    "NIST Machine Translation Evaluation",
    "Amazon Alexa Prize Datasets",
    "Cornell Natural Language Visual Reasoning Datasets",
    "NLVR",
    "WikiSQL",
    "LAMBADA",
    "Language Model Benchmark",
    "BooksCorpus",
    "CQADupStack",
    "Code Query and Question Duplication in StackOverflow",
    "PAN",
    "Authorship Attribution in Forensic Linguistics",
    "One Billion Word Benchmark",
    "TOEFL Essays",
    "Test of English as a Foreign Language",
    "Federated Learning Datasets",
    "LJSpeech",
    "LibriVox Free Audiobook",
    "LIBSVM Datasets",
    "MovieLens",
    "Scikit-learn Datasets",
    "Dialog Dataset",
    "DailyDialog",
    "DialogWAE"
    "OpenSubtitles",
    "Hugging Face Datasets",
    "CLINC",
    "SuperGLUE",
    "XNLI",
    "Medical Datasets",
    "MIMIC-III",
    "CheXpert",
    "MedNLI",
    "SCOTUS",
    "EUR-Lex", 
    "EUPL",
    "MultiNLI",
    "Quandl",
    "Dow Jones Index",
    "Yahoo Finance",
    "OHDSI", 
    "CDC NHANES", 
    "UCI Health"
]

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
eval_metrics_pattern = [[{"LOWER": y} for y in x.lower().split()] for x in nlp_evaluation_metrics]
nlp_evaluation_tasks_pattern  = [[{"LOWER": y} for y in x.lower().split()] for x in nlp_evaluation_tasks]
hyperparameters_pattern  = [[{"LOWER": y} for y in x.lower().split()] for x in hyperparameters]
nlp_datasets_pattern  = [[{"LOWER": y} for y in x.lower().split()] for x in nlp_datasets]
matcher.add("eval_metrics_pattern", eval_metrics_pattern)
matcher.add("nlp_evaluation_tasks_pattern", nlp_evaluation_tasks_pattern)
matcher.add("hyperparameters_pattern", hyperparameters_pattern)
matcher.add("nlp_datasets_pattern", nlp_datasets_pattern)

label_map = {
   "eval_metrics_pattern":"MetricName",
    "nlp_evaluation_tasks_pattern":"TaskName",
    "hyperparameters_pattern":"HyperparameterName",
    "nlp_datasets_pattern":"DatasetName"
}

def remove_overlapping_spans(predictions):
    outputs = []
    for pred in predictions:
        flag = True
        outputsLoop = outputs[:]
        for output in outputsLoop:
            fromVal, toVal = output["value"]["start"], output["value"]["end"]
            start = pred["value"]["start"]
            end = pred["value"]["end"]
            if pred["value"]["start"] in range(fromVal,toVal+1) or pred["value"]["end"] in range(fromVal,toVal+1):
                if (end-start) > (toVal - fromVal):
                    outputs.remove(output)
                else:
                    flag = False
        if flag == True:
            outputs.append(pred)
    return outputs

def generate_pre_annotations(doc):

    matches = matcher(doc)
    predictions = []
    
    
    for match_id, start, end in matches:
        span = doc[start:end]  
        string_id = nlp.vocab.strings[match_id]
        span_label = label_map[string_id]
        predictions.append({
            "id":str(uuid.uuid4()),
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {
              "start": span.start_char,
              "end": span.end_char,
              "text": span.text,
              "labels": [
                span_label
              ]
            }
          }
        )

    return remove_overlapping_spans(predictions)

# if __name__ == "__main__":
#     a=5




