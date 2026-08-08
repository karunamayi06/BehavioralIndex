from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    split, col, when, avg, lower
)

spark = SparkSession.builder \
    .appName("IndexComputation") \
    .master("local[*]") \
    .config("spark.ui.enabled", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ============================================================
# Read CSV files
# ============================================================
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("usage_part_*.csv")

print("\\n=== ORIGINAL DATA ===")
df.show(5, truncate=False)
df.printSchema()


# ============================================================
# 1. Split age_sexe into age and gender
# ============================================================
df = df.withColumn(
    "age",
    split(col("age_sexe"), "-").getItem(0).cast("int")
).withColumn(
    "gender",
    split(col("age_sexe"), "-").getItem(1)
)

print("\\n=== AFTER SPLIT ===")
df.select("age_sexe", "age", "gender").show(10, truncate=False)
df.printSchema()


# ============================================================
# 2. Clean gender values
# ============================================================
df = df.withColumn(
    "gender",
    when(lower(col("gender")).isin("m", "male", "man"), "M")
    .when(lower(col("gender")).isin("f", "female", "woman"), "F")
    .otherwise("Unknown")
)

print("\\n=== AFTER GENDER CLEANING ===")
df.select("age_sexe", "age", "gender").show(10, truncate=False)


# ============================================================
# 3. Create age ranges
# ============================================================
df = df.withColumn(
    "age_range",
    when(col("age") < 18, "0-17")
    .when((col("age") >= 18) & (col("age") <= 24), "18-24")
    .when((col("age") >= 25) & (col("age") <= 34), "25-34")
    .when((col("age") >= 35) & (col("age") <= 44), "35-44")
    .otherwise("45+")
)

print("\\n=== AFTER AGE RANGE ===")
df.select("age", "age_range").show(10, truncate=False)


# ============================================================
# 4. Aggregate metrics
# ============================================================
agg_df = df.groupBy(
    "timestamp",
    "gender",
    "age_range",
    "application"
).agg(
    avg("time_spent").alias("avg_time_spent"),
    avg("times_opened").alias("avg_times_opened"),
    avg("notifications_received").alias("avg_notifications_received"),
    avg("times_opened_after_notification").alias("avg_opened_after_notification")
)

print("\\n=== AGGREGATED DATA ===")
agg_df.show(20, truncate=False)
agg_df.printSchema()


# ============================================================
# 5. Join application categories
# ============================================================
categories_df = spark.read \
    .option("header", "true") \
    .option("sep", "\\t") \
    .option("inferSchema", "true") \
    .csv("applications.csv")

print("\\n=== APPLICATION CATEGORIES ===")
categories_df.show(5, truncate=False)

final_df = agg_df.join(categories_df, on="application", how="left")

print("\\n=== FINAL DATA ===")
final_df.show(20, truncate=False)
final_df.printSchema()

print(f"\\nFinal row count: {final_df.count()}")

spark.stop()