This project uses a neural network to classify the songs in the Free Music Archive (https://github.com/mdeff/fma) based on genre.

This project is originally for a linear algebra class, to demonstrate the effectiveness of PCA on audio data in machine learning.
Later additions have instead focused on achieving higher accuracy with a convolutional neural network, leaving PCA behind.


The expected project structure for data is:
./data/fma-data/fma_[small/medium/large].zip
./data/fma-data/fma_metadata/...

Any alterations to that structure will require minor changes to the code.

Before the cnn can be trained, the melspectrograms must be computed with extract_mel_medium.py. This may take a very long time; expect it to run for at least 5 hours.

Current best accuracy is 60% on 4 genres, trained for 5 hours.
