# CS561Proj

This is the CS561 group project constructed by Andrew Rossman, , and to address the final project assignment of applying NLP to a real world problem.

# Setup

To use these files, download the dependencies to your environment as specified in the environment file, "dependencies.env". 

# Abstract

Health communication in healthcare is a crucial part in the care healthcare process. However, consultation or gaining trueful and accurate medical information from medical professionals can prove difficult with a lack of time or money. A simple search online could yield misleading information or may not yield answers in a timely fashion depending on the severity of the case. 

Additionally, giving potentially false information during a medical crisis becomes a financial liability for any harm done to the patient. MEDiFriend is a medical chatbot utilizing the latest in Natural Language Processing that takes a hybrid 2 step approach to diagnose the situation, provide basic details about proper care before consulting a doctor, and then follow up with info verification to check for potentially false information. 

The chatbot utilizes a transformer model pre-trained on a bespoke dataset. This toy dataset, although smaller in scale, concatenates a wide range of medical terminologies and scenarios to ensure a comprehensive understanding of medical inquiries. A primary focus of our application is the two-stage response validation mechanism: The first stage involves finding the context that fits the response, focusing on basic accuracy and relevance, the second stage introduces an innovative use of foundational models, serving as a sophisticated benchmark for response validation,verifying whether the response can satisfy the prompt. For the question presented to the MEDiFriend bot that is not understood or is not present in the database, the question will be stored in the database and handed out to the expert.

# How to Run

Use the python notebook file inside called Verte inside of src

# Datasets
MEDIQA2019 - https://github.com/abachaa/MEDIQA2019/tree/master

LiveQA_MedicalTask_TREC2017 - https://github.com/abachaa/LiveQA_MedicalTask_TREC2017

Medical dataset for NLP problem - https://www.kaggle.com/datasets/xhlulu/medal-emnlp


Healthcare NLP: LLMs, Transformers, Datasets - https://www.kaggle.com/datasets/jpmiller/layoutlm

National Survey of Residential Care Facilities - https://datasetdirectory.disabilitystatistics.org/show/19

Spinal Cord Injury Model System - https://datasetdirectory.disabilitystatistics.org/show/109


