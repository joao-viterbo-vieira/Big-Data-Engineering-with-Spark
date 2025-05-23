# NYC Yellow Taxi Big Data Engineering Project

## Project Overview
This project analyzes the NYC Yellow Taxi dataset from 2023 to 2025 using Spark and other big data technologies. It involves comprehensive data cleaning, exploratory analysis to uncover meaningful insights and patterns, the development of a data warehouse built on a star schema, and the application of machine learning techniques to predict various aspects of taxi trips. Additionally, the solution was developed on Google Cloud Dataproc utilizing cluster parallelization to assess the advantages of cloud computing, resulting in enhanced performance and more scalable data processing.

## Data Source
The dataset includes NYC yellow taxi trip records from January 2023 to February 2025, provided as Parquet files by the NYC Taxi & Limousine Commission (TLC). Each file contains information about pickup/dropoff times and locations, trip distances, fares, passenger counts, and payment methods.

Data source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

## Dataset Profiling
The raw dataset comprises 26 Parquet files totaling 1.45 GB on disk. Through PySpark profiling, we determined it contains approximately 86 million records (rows) and 22 columns (features), including temporal, geographic, and fare-related attributes. Summary statistics, distribution analyses, and missing value assessments were generated for all features to evaluate data quality and guide subsequent cleaning and feature engineering.

## Project Structure

### Notebooks
- **Descriptive&Pattern_Analysis.ipynb**: Comprehensive exploratory data analysis and pattern discovery
- **dropoff_location_prediction.ipynb**: Machine learning model to predict dropoff locations
- **taxi_predict_tip.ipynb**: Machine learning model to predict tip amounts
- **total_amount_prediction.ipynb**: Machine learning model to predict total fare amounts
- **taxi_first_steps.ipynb**: Initial data exploration and preprocessing steps and prediction of fare amount
- **testing_pandas.ipynb**: Testing notebook for Pandas operations
- **DW_&_ETL.ipynb**: Main notebook for Data Warehouse creation using a star schema design with fact and dimension tables
- **nyc_pipeline_cloud.py**: Example of the code used in Google Dataproc for machine learning implementation and time testing
- **G2_code_examplecloud.py**: Example of the code used in Google Dataproc for machine learning implementation and time testing

### Report and Presentation
- **G2_report.pdf**: Project report
- **G2_presentation.pdf**: Project Presentation
- **spark_ml_training_times.xlsx**: Results of ML training in Google Dataproc

### Data Files
- **data/**: Directory containing yellow taxi data in Parquet format (all months from 2023-01 to 2025-02)
- **data_dictionary_trip_records_yellow.pdf**: Documentation of all fields in the dataset
- **taxi_zone_lookup.csv**: Lookup table mapping location IDs to zone names and boroughs

## Key Features

### Descriptive Analysis and Pattern Analysis (Descriptive&Pattern_Analysis.ipynb)
- Temporal patterns analysis (hourly, daily, monthly trends)
- Geographic hotspot identification (popular pickup/dropoff locations)
- Trip metrics analysis (distance, duration, fare)
- Payment method analysis
- Correlation analysis between different variables

### Predictive Models
- **Tip prediction**: Predicts the tip amount based on trip characteristics
- **Dropoff location prediction**: Predicts likely dropoff locations based on pickup location and other features
- **Total amount prediction**: Forecasts the total fare amount based on trip features

### Data Warehouse & ETL (DW_&_ETL.ipynb)
- Processing of large-scale taxi data (26+ months) using PySpark
- Data cleaning and missing value handling
- Feature engineering including temporal and geographic features
- Implementation of a star schema with:
  - Fact table: Trip records with key metrics
  - Dimension tables: Time, Location, Vendor, Payment Type, Rate Code, etc.
- Data quality validation and exploration

## Technologies Used
- **PySpark**: For big data processing and analysis
- **Python**: Core programming language
- **Matplotlib/Seaborn**: For data visualization
- **SparkSQL**: SparkSQL for data querying
- **Dataframe API**: Dataframe API for data querying
- **Machine Learning**: Various algorithms for predictive modeling (Spark ML)
- **Google Cloud Computing**: Used for testing spark paralelization in the cloud

## How to Use
1. Follow the rules presented in Moodle

## Data (1.43Gb) 
Download the [data](https://github.com/joao-viterbo-vieira/Big-Data-Engineering-with-Spark) folder from the GitHub repository to obtain the 26 Parquet files totaling 1.43Gb. (link:https://github.com/joao-viterbo-vieira/Big-Data-Engineering-with-Spark)


## Results 
The project provides insights into NYC taxi patterns, including:
- Peak hours and days for taxi services
- Most popular routes and zones
- Factors affecting fare amounts and tips
- Prediction models for various trip outcomes
- A comprehensive data warehouse for ongoing analysis
- Analysis of spark parameteres impact in model trainning

## Contributors
- Group 2 Big Data Project Team:
- João Matos	
- Mihai-Adrian Cican	
- Francisco Pinto	
- Manuel Silva	
- João Soares	
- João Vieira

