# prediction-extraction
This was the code I wrote for an independent study project I did in the CIS department (CIS 099). The writeup for the project is [`automated-identification-extraction.pdf`](https://github.com/Chanlaw/prediction-extraction/blob/master/report/automated-identification-extraction.pdf). 

## Webscraping 
First, navigate to the location of `web_crawler` on your computer. 
```
cd location/on/your/computer/prediction-extraction/web_crawler
```
Next, use `python_crawler.py` and `get_clean_text.py` to get the text from Paul Krugman's New York Times Column. 
```
python python_crawler.py > krugman_url.txt
cat krugman_url.txt | python get_clean_text.py > krugman_articles.txt
```
Finally, use `sentence_parser.py` to segment the articles into sentences.
```
cat krugman_articles.txt | python sentence_parser.py > krugman_sentences.tsv
```
The reason we use `.tsv` is because Crowdflower does not support using `.txt` files for uploading data.

## Machine Learning Classifier
To train a classifier, simply run the corresponding file, located in the `classifiers` subdirectory, in python. For example, to train a logistic regression classifier, simply run:
```
python logistic_classifier.py
```
To change the parameters of the model, simply edit the corresponding parts of the python code. 
