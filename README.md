# Disaster Response Pipeline Project

### Motivation
Motivation of this project is to classify text messages during a disaster to make them reachable by responsible parties. 

### Installation
Please use virtualenv and after you can install necessary libraries using requirements.txt

`pip install -r /path/to/requirements.txt`

### File Descriptions

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Acknowledgements and Licensing
This project is developed to fulfil the requirements of Udacity Data Scientist Nanodegree. You can use any part of the code in anywhere without any permission to do anything.
