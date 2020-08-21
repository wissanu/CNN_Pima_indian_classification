# CNN_Pima_indian_classification

Classify Pima indian dataset with Keras (testing for comparing extracting information with informal method using CNN as tested)

pre-processing
------------------
- mean impution on each category
- BaggedLOF
- SMOTE

Model-pre-process_input_output
------------------
CNN - Deep learning needed input into format as (instance, row, col, channels)

- padding (length = 9)
- reshape (max_lenght, 3, 3, 1)
- transform output into category.

Model-CNN
-------------------

- first layer : input_size = (3,3,1)
- second layer : Convolunation 2 dimension with 64 filters and 2 kernel
- thrid layer : Convolunation 2 dimension with 32 filters and 2 kernel
- fourth layer : Max pool size = 1
- Flatten
- create 2 hidden layer with 100 nodes (fully connected)
- output 2


The model is interesting in the coverage of learning rate.

Need to be improve.



