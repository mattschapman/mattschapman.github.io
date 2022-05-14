## Selected projects in data science, machine learning and NLP

---

### The Fragile Families Challenge
The Fragile Families Challenge was a predictive modelling challenge commissioned by researchers at Princeton University in 2017. In the challenge, participants were tasked with predicting six life outcomes (GPA, material hardship, grit, eviction, layoff, and job training) for 4,242 children based on their cirumstances between birth and age 9. 

I took part in this challenge, using various ML and NLP techniques including: (i) imputing missing values using word embeddings and KNN, (ii) modelling with LASSO, Random Forests and XGBoost models, (iii) Bayesian hyperparameter optimisation, and (iv) using feature importance scores to interpret the models' predictions. 

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/Jupyter-white?logo=Jupyter)](#) [![](https://img.shields.io/badge/sklearn-white?logo=scikit-learn)](#) [![](https://img.shields.io/badge/Scikit%20Optimize-white?logo=data:image/png;base64,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)](#)

[View code on Colab](/sample_page)

---

### Using BERT and LSTM models for multilabel text classification of ArXiv papers
ArXiv is a popular open-access distribution service for scientific articles. In this project, I compared the performance of LSTM and BERT models for predicting the subject and year of papers given their abstract text.  

[![](https://img.shields.io/badge/Python-white?logo=Python)](#) [![](https://img.shields.io/badge/Jupyter-white?logo=Jupyter)](#) [![](https://img.shields.io/badge/PyTorch-white?logo=pytorch)](#) [![](https://img.shields.io/badge/ArXiv-white?logo=arxiv)](#)

[View code on Colab](/pdf/sample_presentation.pdf)


---

### Implementing a custom data collection pipeline using Scrapy and MongoDB
The web hosts vast quantities of data useful for research. In this project, I built a custom data collection pipeline using the Python library `scrapy` and a MongoDB cluster.

[View code on Github](/pdf/sample_presentation.pdf)

---

### Optimising code with multiprocessing
To overcome the limitations of Python's Global Interpreter Lock (GIL) and speed up the execution of "embarrassingly parallel" problems in Python code, we can run subparts of a progam in parallel (that is, simultaneouly) on multiple CPUs. In this project, I use multiprocessing to count tweets written near London. 

[View code on Github](/pdf/sample_presentation.pdf)

---

### Interpretable machine learning with LIME and Shapley values
Machine learning has enormous potential, yet a significant barrier to its adoption in many fields is the lack of interpretability of many black box models. In this project, I use several techniques to gain insights into models for predicting cervical cancer.  

[View code on Colab](/pdf/sample_presentation.pdf)

---

### Using virtual environments and Geopandas to visualise geospatial data
Virtual environments can be used to create multiple instances of software, which can be used to overcome dependency issues when installing new packages. In this project, I show how a virtual environment can be used to install the Python package `geopandas` and use this to visualise data on deprivation levels in London. 

[View code on Colab](/pdf/sample_presentation.pdf)

---

### Using multi-level modelling in R to investigate the drivers of Covid-19 vaccine hesitancy
In many tasks, failing to account for the hierarchical relations and autocorrelations between data can create "ecological fallacies" which misguide our interpretation of the data. In this project, I use mutli-level regression modelling to account for spatial autocorrelation and study reasons for Covid-19 vaccine hesitancy. 

[View code on Colab](/pdf/sample_presentation.pdf)

---

<!-- [Project 3 Title](http://example.com/)
<img src="images/dummy_thumbnail.jpg?raw=true"/>

---
 -->
## Skills-based projects
A selection of smaller projects demonstrating specific data science and ML skills.

- [Project 1 Title](http://example.com/)
- [Project 2 Title](http://example.com/)
- [Project 3 Title](http://example.com/)
- [Project 4 Title](http://example.com/)
- [Project 5 Title](http://example.com/)



<!-- <p style="font-size:11px">Page template forked from <a href="https://github.com/evanca/quick-portfolio">evanca</a></p> -->
<!-- Remove above link if you don't want to attibute -->
