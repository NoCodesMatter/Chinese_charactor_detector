# data process

In generate error file, there are 4 jupyter file.
generate_error is to generate error data based on correct data.
shape and json process is to transfer .json, file and .txt data.
preprocessing is to turn the data into the format needed.

tools, vocab and words are files used in generate_error.
The data in this file is just sample, not all data.

For data in tests, you need to run generate_error, then preprocessing and json process, it depends on the input.

So, shortly, just run generate_error, the output shows the generation. It can prove the generation.

# model training

in CBZCorrector file run spelling_corrector.py
If the .pth model already exists in the current file directory, the program will not enter the model training process again. 
Instead, we skip the training and go directly to the evaluation section using the existing model

# model evaluation

directly run evaluate.py file, it will show mes and accuracy of model
