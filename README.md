**PREPROCESSING STEPS:**

1. Code to convert PDFs to annotation input format ([/pdf\_to\_text/pdf\_to\_text\_conversion.py](https://github.com/Afreens1997/nlp_from_scratch_assignment/blob/main/pdf_to_text/pdf_to_text_conversion.py))  
  a. Parse PDF to dictionary using scipdf  
  b. Convert scipdf generated dictionary to label studio annotation input format.  
  c. This code includes invocation to utils/preannotation.py  
2. [preannotation.py](https://github.com/Afreens1997/nlp_from_scratch_assignment/blob/main/utils/preannotation.py):
This file includes code to pre-annotate certain entity types (MetricName, DatasetName, HyperparameterName, TaskName), from predefined gazetteers for each of these entity types. We leverage spacy matcher functionality to match the list of entities in gazetteers to spacy doc while tokenizing the text. This gives a list of annotations already pre-populated when we start annotating on label studio.
3. [complete\_annotations.py](https://github.com/Afreens1997/nlp_from_scratch_assignment/blob/main/utils/complete_annotations.py)
This module is used to refine the human annotations, to minimize human errors. We collate the annotations for MethodName, DatasetName, MetricName, TaskName and reannotate via code any missed occurrences of the entities.

**STEPS FOR TRAINING:**

1. Install the required packages with requirements.txt.
2. Run the file with python3 train\_model.py
3. This has the train pipeline embedded in it. (Preprocess, Train)
4. It takes train.conll, test.conll and prepares a directory with the saved checkpoints.
5. To run inference, we need to run python3 inference.py which will create a results.txt. We need to give the model checkpoint and test file paths in the code.
6. After this, we need to run the following two commands:  
  a. python3 transform.py  
  b. python3 assert\_scr.py  
7. We attach the IBM dataset as well. To train on that, just replace the train and test file names, and there is code for finetuning this model on our annotated data.
