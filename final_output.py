from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, TimestampType
)
from pyspark.sql.functions import (
    col, split, trim, upper,
    when, avg, to_date, lit,
    broadcast, lag,
    year, month, dayofmonth
)
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("IndexComputation") \
    .master("local[*]") \
    .config("spark.sql.adaptive.enabled",    "false") \
    .config("spark.sql.shuffle.partitions",  "8") \
    .config("spark.sql.session.timeZone",    "Europe/Paris") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

schema = StructType([
    StructField("timestamp",                       TimestampType(), True),
    StructField("user_id",                         IntegerType(),   True),
    StructField("age_sexe",                        StringType(),    True),
    StructField("application",                     StringType(),    True),
    StructField("time_spent",                      IntegerType(),   True),
    StructField("times_opened",                    IntegerType(),   True),
    StructField("notifications_received",          IntegerType(),   True),
    StructField("times_opened_after_notification", IntegerType(),   True),
])

df=spark.read\
.option("header","true")\
.option("inferSchema","true")\
.csv("usage_part_*.csv")

df = df \
    .withColumn("age",    split(col("age_sexe"), "-").getItem(0).cast(IntegerType())) \
    .withColumn("gender", split(col("age_sexe"), "-").getItem(1))

# ── Step 2: Harmonize gender values ──────────────────────────
df = df.withColumn(
    "gender",
    when(upper(trim(col("gender"))).isin("H", "M", "HOMME"), "M")
   .when(upper(trim(col("gender"))).isin("F", "W", "FEMME"), "F")
   .otherwise(None)
)

df = df.withColumn(
    "age_range",
    when(col("age") < 18,              "0-17")
   .when(col("age").between(18, 24),   "18-24")
   .when(col("age").between(25, 34),   "25-34")
   .when(col("age").between(35, 44),   "35-44")
   .when(col("age").between(45, 54),   "45-54")
   .otherwise("55+")
)

df_agg = df.groupBy(
        "timestamp", "gender", "age_range", "application"
    ).agg(
        avg("time_spent")                     .alias("avg_time_spent"),
        avg("times_opened")                   .alias("avg_times_opened"),
        avg("notifications_received")         .alias("avg_notifications_received"),
        avg("times_opened_after_notification").alias("avg_times_opened_after_notification"),
    )

cat_schema = StructType([
    StructField("application", StringType(), True),
    StructField("category",    StringType(), True),
])

df_categories = spark.read \
    .option("header", "true") \
    .schema(cat_schema) \
    .option("sep", "\\t") \
    .csv("applications.csv")

df_enriched = df_agg.join(
    broadcast(df_categories),
    on="application",
    how="left"
)

df=df_enriched.withColumn("date",to_date(col("timestamp")))


df_gender=df.groupBy("date","gender").agg(
    avg("avg_time_spent").alias("value")
).withColumn(
    "criterion",
    lit("gender")
).withColumnRenamed(
    "gender","variable"
)
df_gender.show(10)

df_age = df.groupBy("date", "age_range").agg(
    avg("avg_time_spent").alias("value")
).withColumn(
    "criterion",
    lit("age_range")
).withColumnRenamed(
    "age_range",
    "variable"
)

print("=== STEP 3: BY AGE RANGE ===")
df_age.orderBy("date", "variable").show(10, truncate=False)

df_category = df.groupBy("date", "category").agg(
    avg("avg_time_spent").alias("value")
).withColumn(
    "criterion",
    lit("category")
).withColumnRenamed(
    "category",
    "variable"
)

print("=== STEP 4: BY CATEGORY ===")
df_category.orderBy("date", "variable").show(10, truncate=False)

df_long = df_gender.unionByName(df_age).unionByName(df_category)

print("=== STEP 5: LONG FORMAT ===")
df_long.orderBy("date", "criterion", "variable").show(15, truncate=False)

df_long = df_long.withColumn("year", year(col("date"))) \
                 .withColumn("month", month(col("date"))) \
                 .withColumn("day", dayofmonth(col("date")))

w_index = Window.partitionBy(
    "criterion",
    "variable",
    "month",
    "day"
).orderBy("year")

df_long = df_long.withColumn(
    "value_1yr_ago",
    lag("value", 1).over(w_index)
)

df_long = df_long.withColumn(
    "index",
    col("value") / col("value_1yr_ago")
)
#average of 5 days
w_smooth = Window.partitionBy(
    "criterion",
    "variable"
).orderBy("date").rowsBetween(-4, 0)
# removes daily spikes by averaging the index over a 5-day window
df_long = df_long.withColumn(
    "smoothed_index",
    avg("index").over(w_smooth)
).drop("year", "month", "day")

print("=== FINAL OUTPUT ===")
df_long.orderBy("date", "criterion", "variable").show(20, truncate=False)

print("Total rows:", df_long.count())

output_path="final_output.csv"

df_long.repartition("criterion")\
.write\
.mode("overwrite")\
.partitionBy("criterion")\
.parquet(output_path)

df_check=spark.read.parquet(output_path)

print(f"\nRow count: {df_check.count()}")
print(f"\nSchema:")
df_check.printSchema()
 
print(f"\nPartition values (criterion):")
df_check.select("criterion").distinct().show()
 
print(f"\nSample rows:")
df_check.show(5, truncate=False)

# ── Keep Spark alive for UI inspection ───────────────────────
input("\nSpark UI: localhost:4040  —  press Enter to stop.\n")
spark.stop()

