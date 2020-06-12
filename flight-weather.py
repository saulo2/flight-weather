#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark import SparkConf

conf = SparkConf()     .setAppName("flight-weather")     .set("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")     .set("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc1")


# In[2]:


from pyspark.sql import SparkSession

# Spark session & context
spark = SparkSession.builder.master('local').config(conf=conf).getOrCreate()
sc = spark.sparkContext

# Sum of the first 100 whole numbers
rdd = sc.parallelize(range(100 + 1))
rdd.sum()
# 5050


# In[3]:


from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)


# In[4]:


# Read dataset
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, TimestampType
df = (sqlContext.read.format("csv").
  option("header", "true").
  option("nullValue", "NA").
  option("inferSchema", True).
  load("/data/flight-weather.csv"))


# In[5]:


# ARR_DEL15 = 1 if it's canceled.
from pyspark.sql.functions import when
df = df.withColumn("ARR_DEL15", when(df["CANCELLED"] == 1, 1).otherwise(df["ARR_DEL15"]))


# In[6]:


# Remove flights if it's diverted.
df = df.filter(df["DIVERTED"] == 0)


# In[7]:


# Select required columns
df = df.select(
  "ARR_DEL15",
  "MONTH",
  "DAY_OF_WEEK",
  "UNIQUE_CARRIER",
  "ORIGIN",
  "DEST",
  "CRS_DEP_TIME",
  "CRS_ARR_TIME",
  "RelativeHumidityOrigin",
  "AltimeterOrigin",
  "DryBulbCelsiusOrigin",
  "WindSpeedOrigin",
  "VisibilityOrigin",
  "DewPointCelsiusOrigin",
  "RelativeHumidityDest",
  "AltimeterDest",
  "DryBulbCelsiusDest",
  "WindSpeedDest",
  "VisibilityDest",
  "DewPointCelsiusDest")


# In[8]:


# Drop rows with null values
df = df.dropna()


# In[9]:


# Convert categorical values to indexer (0, 1, ...)
from pyspark.ml.feature import StringIndexer
uniqueCarrierIndexer = StringIndexer(inputCol="UNIQUE_CARRIER", outputCol="Indexed_UNIQUE_CARRIER").fit(df)
originIndexer = StringIndexer(inputCol="ORIGIN", outputCol="Indexed_ORIGIN").fit(df)
destIndexer = StringIndexer(inputCol="DEST", outputCol="Indexed_DEST").fit(df)
arrDel15Indexer = StringIndexer(inputCol="ARR_DEL15", outputCol="Indexed_ARR_DEL15").fit(df)


# In[10]:


# Assemble feature columns
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(
  inputCols = [
    "MONTH",
    "DAY_OF_WEEK",
    "Indexed_UNIQUE_CARRIER",
    "Indexed_ORIGIN",
    "Indexed_DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "RelativeHumidityOrigin",
    "AltimeterOrigin",
    "DryBulbCelsiusOrigin",
    "WindSpeedOrigin",
    "VisibilityOrigin",
    "DewPointCelsiusOrigin",
    "RelativeHumidityDest",
    "AltimeterDest",
    "DryBulbCelsiusDest",
    "WindSpeedDest",
    "VisibilityDest",
    "DewPointCelsiusDest"],
  outputCol = "features")


# In[11]:


#get_ipython().system('pip install mmlspark')


# In[12]:


import pyspark
spark = pyspark.sql.SparkSession.builder.appName("MyApp")    .config("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc1")    .getOrCreate()
import mmlspark


# In[13]:


# Define classifier
from mmlspark.lightgbm import LightGBMClassifier
classifier = LightGBMClassifier(
  featuresCol="features",
  labelCol="ARR_DEL15",
  numIterations=150)


# In[14]:


# Create pipeline
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[uniqueCarrierIndexer, originIndexer, destIndexer, arrDel15Indexer, assembler, classifier])


# In[17]:


# Prepare training with ParamGridBuilder
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
### 4 x 2 = 8 times training occurs
paramGrid = ParamGridBuilder()   .addGrid(classifier.learningRate, [0.1])   .addGrid(classifier.numLeaves, [100])   .build()
   #.addGrid(classifier.learningRate, [0.1, 0.3, 0.5, 0.7]) \
   #.addGrid(classifier.numLeaves, [100, 200]) \
tvs = TrainValidationSplit(
  estimator=pipeline,
  estimatorParamMaps=paramGrid,
  evaluator=MulticlassClassificationEvaluator(labelCol="ARR_DEL15", predictionCol="prediction", metricName="weightedPrecision"),
  trainRatio=0.8)  # data is separated by 80% and 20%, in which the former is used for training and the latter for evaluation


# In[18]:


#get_ipython().system('pip install mlflow')


# In[19]:


# Start mlflow and run pipeline
import mlflow
from mlflow import spark
with mlflow.start_run():
  model = tvs.fit(df)                                               # logs above metric (weightedPrecision)
  mlflow.log_metric("bestPrecision", max(model.validationMetrics)); # logs user-defined custom metric
  mlflow.spark.log_model(model.bestModel, "model-file")             # logs model as artifacts


# In[ ]:




