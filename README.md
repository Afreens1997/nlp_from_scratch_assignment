* Install the required packages with requirements.txt.
* Run the file with python3 train_model.py 
* This has the train pipeline embedded in it. (Preprocess, Train)
* It takes train.conll, test.conll and prepares a directory with the saved checkpoints.
* To run inference, we need to run python3 inference.py which will create a results.txt. We need to give the model checkpoint and test file paths in the code.
* After this, we need to run the following two commands:
	a. python3 transform.py
	b. python3 assert_scr.py
* We attach the IBM dataset as well. To train on that, just replace the train and test file names, and there is code for finetuning this model on our annotated data.