from datetime import datetime, timezone
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, IntegerType, LongType, ShortType, DoubleType, FloatType, BooleanType, DateType, TimestampType, BinaryType, StructField, ArrayType
from pyspark.sql.functions import (count, when, isnull, col, lit, element_at, coalesce, cos, radians, round as spark_round, xxhash64, pmod, desc)

# Columns read from the building files. geometry (WKB) is the heaviest column and it is not needed for the analysis, so it is left out 
BUILDING_COLUMNS = ["id", "subtype", "class", "height", "num_floors", "is_underground", "has_parts", "sources", "bbox"]
RESIDENTIAL_CLASSES = ["house", "residential", "apartments", "detached", "semidetached_house", "terrace", "bungalow"]
UNCLASSIFIED = "unclassified"

# Approximate conversion factors from degrees to meters (WGS84)
METERS_PER_DEG_LAT = 110540.0
METERS_PER_DEG_LON = 111320.0


class SparkUtils:

    def __init__(self, app_name, master_url=None, spark_jars=None, spark_packages=None):
        spark_builder = (
            SparkSession.builder
            .appName(app_name))

        if master_url is not None:
            spark_builder = spark_builder.master(master_url)

        if spark_jars is not None:
            spark_builder = spark_builder.config("spark.jars", spark_jars)

        if spark_packages is not None:
            spark_builder = spark_builder.config("spark.jars.packages", spark_packages)

        self._spark = spark_builder.getOrCreate()

    @property
    def spark(self):
        return self._spark

    @staticmethod
    def generate_schema(columns_info) -> StructType:
        """
        Generates a list of StructField objects from a list of tuples.

        Args:
            column_info (list of tuples): Each tuple contains (column_name, data_type_string).

        Returns:
            list: A list of StructField objects.
        """
        # Mapping from string type names to PySpark data types
        type_mapping = {
            "string": StringType(),
            "int": IntegerType(),
            "long": LongType(),
            "short": ShortType(),
            "double": DoubleType(),
            "float": FloatType(),
            "boolean": BooleanType(),
            "date": DateType(),
            "timestamp": TimestampType(),
            "binary": BinaryType(),
            "array_int": ArrayType(IntegerType()),
            "array_string": ArrayType(StringType()),
            "struct": StructType()
        }

        struct_fields = []
        for column_info in columns_info:
            col_name  = column_info[0]
            col_type  = column_info[1]

            if col_type == "struct":
                # Expect a third element with the sub-field definitions
                if len(column_info) < 3 or not isinstance(column_info[2], list):
                    raise ValueError(
                        f"Column '{col_name}' is declared as 'struct' but "
                        f"no sub-fields list was provided as the third element."
                    )
                # Recurse to build the nested StructType
                nested_schema = SparkUtils.generate_schema(column_info[2])
                struct_field  = StructField(col_name, nested_schema, True)

            elif col_type not in type_mapping:
                raise ValueError(f"Unsupported data type: '{col_type}' for column '{col_name}'")

            else:
                # Create a StructField for the column
                struct_field = StructField(column_info[0], type_mapping[column_info[1]], True)

            struct_fields.append(struct_field)

        return StructType(struct_fields)

    @staticmethod
    def count_nulls(df):
        return df.select([count(when(isnull(c), c)).alias(c) for c in df.columns])
 
    # Reading 
    def read_buildings(self, path, sample_fraction=None):
        """Reads the Overture building Parquet files keeping only the columns used"""
        df = self._spark.read.parquet(path).select(*BUILDING_COLUMNS)
        if sample_fraction:
            df = df.sample(fraction=sample_fraction, seed=42)
        return df
 
    # Data cleaning 
    @staticmethod
    def null_percentages(df):
        """Percentage of nulls per column, one row. Nested columns are checked as a whole"""
        total = count(lit(1))
        return df.select([
            spark_round(100.0 * count(when(isnull(c), lit(1))) / total, 2).alias(c)
            for c in df.columns
        ])

    @staticmethod
    def clean_buildings(df):
        """ Drops records without id or bbox and records with a malformed bbox
        Drops duplicated ids
        Turns out-of-range heights/floors into nulls
        Fills null subtype / class with 'unclassified'
        The sources array is reduced to its first dataset before dropDuplicates, so the shuffle only moves small flat columns instead of the whole nested array."""
        valid_bbox = (
            col("bbox.xmin").isNotNull() & col("bbox.ymin").isNotNull()
            & (col("bbox.xmin") <= col("bbox.xmax")) & (col("bbox.ymin") <= col("bbox.ymax"))
            & col("bbox.xmin").between(-180, 180) & col("bbox.xmax").between(-180, 180)
            & col("bbox.ymin").between(-90, 90) & col("bbox.ymax").between(-90, 90)
        )
        return (
            df.filter(col("id").isNotNull() & valid_bbox)
            .withColumn("primary_source", coalesce(element_at(col("sources"), 1).getField("dataset"),
                                                   lit("unknown")))
            .drop("sources")
            .dropDuplicates(["id"])
            .withColumn("height", when(col("height").between(0, 1000), col("height")))
            .withColumn("num_floors", when(col("num_floors").between(0, 200), col("num_floors")))
            .withColumn("is_underground", coalesce(col("is_underground"), lit(False)))
            .fillna(UNCLASSIFIED, subset=["subtype", "class"])
        )
 
    # Column derivation 
    @staticmethod
    def region_expr(lon, lat):
        """ Approximate macro-region of the Americas from a lon/lat point, using bounding boxes (not official borders). The order of the conditions matters. """
        # Approximate northern border of Mexico 
        mexico_north_lat = (
            when(lon < -114.8, lit(32.7))
            .when(lon < -106.5, lit(31.4))
            .when(lon < -97.1, lit(31.8) + (lon + 106.5) * ((25.9 - 31.8) / 9.4))  # Rio Bravo
            .otherwise(lit(25.9))
        )
        # Pacific islands 
        pacific_islands = (lon < -118.5) & (lat < 32.5)
        return (
            when((lon > -30) | (lon < -170) | pacific_islands, lit("other"))
            .when((lat < 7.2) | ((lat < 12.5) & (lon > -77.2)), lit("south_america"))
            .when((lat < 18) & (lon > -92.2), lit("central_america"))
            .when(((lat >= 17.5) & (lat < 24) & (lon > -85))
                  | ((lat >= 10) & (lat < 17.5) & (lon > -68)), lit("caribbean"))
            .when((lat >= 14.5) & (lon < -86.7) & (lat < mexico_north_lat), lit("mexico"))
            .when(lat >= 24, lit("us_canada"))
            .otherwise(lit("other"))
        )

    @staticmethod
    def add_derived_columns(df):
        """Adds centroid, approximate area, completeness flags, size, region and housing type (primary_source is derived in clean_buildings)."""
        centroid_lon = (col("bbox.xmin") + col("bbox.xmax")) / 2
        centroid_lat = (col("bbox.ymin") + col("bbox.ymax")) / 2

        df = (
            df.withColumn("centroid_lon", spark_round(centroid_lon, 6))
            .withColumn("centroid_lat", spark_round(centroid_lat, 6))
            .withColumn("area_bbox_m2", spark_round(
                (col("bbox.xmax") - col("bbox.xmin")) * lit(METERS_PER_DEG_LON) * cos(radians(centroid_lat))
                * (col("bbox.ymax") - col("bbox.ymin")) * lit(METERS_PER_DEG_LAT), 2))
            .withColumn("has_height", col("height").isNotNull())
            .withColumn("has_floors", col("num_floors").isNotNull())
            .withColumn("has_class", col("class") != UNCLASSIFIED)
            .withColumn("region", SparkUtils.region_expr(centroid_lon, centroid_lat))
        )

        df = df.withColumn(
            "size_category",
            when(col("area_bbox_m2") < 60, lit("small"))
            .when(col("area_bbox_m2") < 250, lit("medium"))
            .when(col("area_bbox_m2") < 1000, lit("large"))
            .otherwise(lit("very_large"))
        )

        is_declared_residential = (col("subtype") == "residential") | col("class").isin(RESIDENTIAL_CLASSES)
        is_probable_house = (
            (col("subtype") == UNCLASSIFIED)
            & col("area_bbox_m2").between(25, 400)
            & (col("height").isNull() | (col("height") <= 12))
        )
        return df.withColumn(
            "housing_type",
            when(is_declared_residential, lit("declared_residential"))
            .when(is_probable_house, lit("probable_housing"))
            .when(col("subtype") != UNCLASSIFIED, lit("non_residential"))
            .otherwise(lit("undetermined"))
        )
 
    # Filtering and sorting 
    @staticmethod
    def filter_relevant(df):
        """Keeps above-ground buildings with a posible footprint (10 m2 .. 1 km2) inside the Americas  """
        return df.filter(
            (~col("is_underground"))
            & col("area_bbox_m2").between(10, 1_000_000)
            & (col("region") != "other")
        )

    @staticmethod
    def select_output_columns(df):
        """Flat columns that are persisted  """
        return df.select( "id", "region", "housing_type", "subtype", "class", "primary_source", "centroid_lon", "centroid_lat", "area_bbox_m2", "size_category", "height", "num_floors", "has_height", "has_floors", "has_class", "has_parts" )

    @staticmethod
    def sort_by_size(df):
        """Sorts buildings from the largest to the smallest footprint (ties by id) """
        return df.orderBy(desc("area_bbox_m2"), col("id"))
 
    # Persistence 
    @staticmethod
    def persist(df):
        return df.persist(StorageLevel.MEMORY_AND_DISK)

    @staticmethod
    def write_partitioned(df, path, partition_cols, files_per_partition=8):
        """ Writes Parquet partitioned by partition_cols. Rows are spread by partition columns plus a
        hash bucket of id, so big partitions are written by several tasks (avoids skew) while
        each partition folder keeps at most files_per_partition files. """

        bucket = pmod(xxhash64(col("id")), lit(files_per_partition))
        (df.repartition(*[col(c) for c in partition_cols], bucket)
            .write.mode("overwrite")
            .partitionBy(*partition_cols)
            .parquet(path))
 
    # ------------------------------------------------------------------
    # Debugging helpers (useful on EMR, where stdout is buried in logs)
    # ------------------------------------------------------------------
    
    @staticmethod
    def schema_as_text(df) -> str:
        """Returns the same tree that df.printSchema() prints, as a string."""
        return df._jdf.schema().treeString()

    @staticmethod
    def sample_as_text(df, rows=20, truncate=80, vertical=True) -> str:
        """Returns the same table that df.show() prints, as a string."""
        return df._jdf.showString(rows, truncate, vertical)

    @staticmethod
    def sample_as_json(df, rows=20) -> list:
        """Returns the first rows as JSON strings (keeps nested structs readable)."""
        return df.limit(rows).toJSON().collect()

    def debug_report(self, df, title="", rows=20, count_rows=False) -> str:
        """Builds a text report with input files, schema and a sample of records."""
        sample = df.limit(rows)
        files = df.inputFiles()

        lines = [f"Debug report - {datetime.now(timezone.utc).isoformat()}", title, ""]

        lines.append(f"Files read: {len(files)}")
        for f in files[:10]:
            lines.append(f"  {f}")
        if len(files) > 10:
            lines.append(f"  ... and {len(files) - 10} more")
        lines.append(f"Columns: {len(df.columns)}")
        if count_rows:
            lines.append(f"Row count: {df.count():,}")  # full scan
        lines.append("")

        lines.append("=== SCHEMA ===")
        lines.append(SparkUtils.schema_as_text(df))

        lines.append(f"=== SAMPLE ({rows} rows, vertical) ===")
        lines.append(SparkUtils.sample_as_text(sample, rows))

        lines.append(f"=== SAMPLE ({rows} rows, JSON) ===")
        lines.extend(SparkUtils.sample_as_json(sample, rows))

        return "\n".join(lines) + "\n"

    def pipeline_report(self, df_raw, df_final, df_sorted, title="", rows=5, count_rows=False, null_stats=True) -> str:
        """Builds a text report of the pipeline: nulls, counts, final schema/sample and largest buildings.
        null_stats=False skips the % of nulls (a full scan of the raw data)."""
        lines = [f"Pipeline report - {datetime.now(timezone.utc).isoformat()}", title, ""]

        lines.append(f"Files read: {len(df_raw.inputFiles())}")
        lines.append("")
        lines.append("=== RAW SCHEMA (selected columns) ===")
        lines.append(SparkUtils.schema_as_text(df_raw))

        if null_stats:
            lines.append("=== % NULLS PER COLUMN (raw) ===")
            lines.append(SparkUtils.sample_as_text(SparkUtils.null_percentages(df_raw), 1, 0))

        if count_rows:
            lines.append(f"Raw rows:   {df_raw.count():,}")
            lines.append(f"Final rows: {df_final.count():,}")
            lines.append("")

        lines.append("=== FINAL SCHEMA ===")
        lines.append(SparkUtils.schema_as_text(df_final))

        lines.append(f"=== FINAL SAMPLE ({rows} rows) ===")
        lines.append(SparkUtils.sample_as_text(df_final, rows))

        lines.append("=== LARGEST BUILDINGS (sorted by area_bbox_m2) ===")
        lines.append(SparkUtils.sample_as_text(df_sorted, 20, 0, False))

        return "\n".join(lines) + "\n"

    def write_text(self, path, text):
        """
        Writes a single text file using Hadoop's FileSystem API.
        Works with s3://, hdfs:// and file:// paths (overwrites if it exists).
        """
        jvm = self._spark._jvm
        hadoop_path = jvm.org.apache.hadoop.fs.Path(path)
        fs = hadoop_path.getFileSystem(self._spark._jsc.hadoopConfiguration())
        stream = fs.create(hadoop_path, True)
        try:
            stream.write(bytearray(text.encode("utf-8")))
        finally:
            stream.close()
