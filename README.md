# Quora Comments Repository

This repository contains the progress that I have made on the Quora Insincere Questions Classification kaggle competition having the following description.

>An existential problem for any major website today is how to handle toxic and divisive content. Quora wants to tackle this problem head-on to keep their platform a place where users can feel safe sharing their knowledge with the world.
>
>Quora is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers.
>
>In this competition, Kagglers will develop models that identify and flag insincere questions. To date, Quora has employed both machine learning and manual review to address this problem. With your help, they can develop more scalable methods to detect toxic and misleading content.
>
>Here's your chance to combat online trolls at scale. Help Quora uphold their policy of “Be Nice, Be Respectful” and continue to be a place for sharing and growing the world’s knowledge.

This repository chiefly three different classifiers

* `neural_bow_classifier.py`: This is a feed forward nueral network developed in Bengio et. al. 2009 in the Journal of Machine Learning Research in the utilizing a set of provided word embeddings. 

* `neural_cnn_classifier.py`: This is convolution neural network developed in Kim, Yoon 2014 arXiv "Convolution Neural Networks for Sentence Classification" also utilizing a set of provided embeddings.

* `quora_ngram.py`: This is an implementation of various traditional smoothed ngram methods, that serve as a good basis of analysis. :w

Please have a look and send any questions or comments to eric.pnr@gmail.com
