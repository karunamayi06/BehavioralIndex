from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,StructField,StringType,IntegerType,TimestampType
)

spark=SparkSession.builder\
.appName("IndexComputation") \
.master("local[*]") \
.getOrCreate()

spark.sparkContext.setLogLevel("WARN")

schema = StructType([
    StructField("timestamp", TimestampType(), True),
    StructField("user_id", IntegerType(), True),
    StructField("age_sexe", StringType(), True),
    StructField("application", StringType(), True),
    StructField("time_spent", IntegerType(), True),
    StructField("times_opened", IntegerType(), True),
    StructField("notifications_received", IntegerType(), True),
    StructField("times_opened_after_notification", IntegerType(), True),

])

df=spark.read\
.option("header","true")\
.option("inferSchema","true")\
.csv("*.csv")

print("\\n---Schema---")
df.printSchema()

print("\\n--Sample rows--")
df.show(5,truncate=False)

print(f"\\n--Total rows:{df.count()}--")

df=df.withColumn("age",s)

spark.stop()