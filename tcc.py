
# Install librarys
import os
import sys
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from sparknlp.annotator import (Tokenizer,WordEmbeddingsModel,Word2VecModel)
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, OneHotEncoder, StringIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

# Start Spark
#spark = sparknlp.start()
spark =SparkSession.builder .master("local[4]") .config("spark.driver.memory", "16G") .config("spark.driver.maxResultSize", "0") .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") .config("spark.kryoserializer.buffer.max", "2000m") .config("spark.jsl.settings.pretrained.cache_folder", "sample_data/pretrained") .config("spark.jsl.settings.storage.cluster_tmp_dir", "sample_data/storage") .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.0.0") .getOrCreate()

print("Spark NLP version: ", sparknlp.version())
print("Apache Spark version: ", spark.version)

spark

# Loading database
base = spark.read \
      .option("header", True) \
      .csv("base3.csv")

base.show(truncate=False);
print (type(base));

# Start TRANSFORMER

documentAssembler = DocumentAssembler().setInputCol("quadri").setOutputCol("document")
doc_df=documentAssembler.transform(base)
doc_df.show()

tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
doc_df2=tokenizer.fit(doc_df).transform(doc_df)
type(doc_df2)
doc_df2.select("token.result").show(truncate=False)

finisher = Finisher() \
      .setInputCols(["token"]) \
      .setOutputCols(["token_features"]) \
      .setOutputAsArray(True) \
      .setCleanAnnotations(False)

# CountVectos is fundamental to learning epoach
countVectors = CountVectorizer(inputCol="token_features", outputCol="features", vocabSize=10000, minDF=5)

from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, OneHotEncoder, StringIndexer, VectorAssembler, SQLTransformer
label_stringIdx = StringIndexer(inputCol = "exp", outputCol = "label")
nlp_pipeline = Pipeline(
    stages=[documentAssembler,
            tokenizer,finisher,countVectors, label_stringIdx])

nlp_model = nlp_pipeline.fit(base)

processed = nlp_model.transform(base)
processed.count()

type(label_stringIdx)

processed.select('*').show(truncate=50)
processed.select('quadri','token').show(truncate=50)

processed.select('token_features').show(truncate=False)

(trainingData, testData) = processed.randomSplit([0.7, 0.3], seed = 100)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))

trainingData.show()

processed.select('label','exp').show()

trainingData.printSchema()

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0)

lrModel = lr.fit(trainingData)

lrn_summary = lrModel.summary
lrn_summary.predictions.show()

predictions = lrModel.transform(testData)

predictions.filter(predictions['prediction'] == 0) \
    .select("quadri","exp","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)

#from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

print(evaluator.evaluate(predictions))

