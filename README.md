# CNN_Pima_indian_classification

Classify Pima indian dataset with Keras (Test for fun)

pre-processing
------------------
1.) mean impution on each category
2.) BaggedLOF
3.) SMOTE

Model-pre-process_input_output
------------------
CNN - Deep learning needed input into format as (instance, row, col, channels)

1.) padding (length = 9)
2.) reshape (max_lenght, 3, 3, 1)
3.) transform output into category.

Model-CNN
-------------------

1.) first layer : input_size = (3,3,1)
2.) second layer : Convolunation 2 dimension with 64 filters and 2 kernel
3.) thrid layer : Convolunation 2 dimension with 32 filters and 2 kernel
4.) fourth layer : Max pool size = 1
5.) Flatten
6.) create 2 hidden layer with 100 nodes (fully connected)
7.) output 2


The model is interesting in the coverage of learning rate.
but it poor cuz "overfitting".

Need to be improve.



