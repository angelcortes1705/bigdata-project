"""
project.py
Batch pipeline. This script only orchestrates the steps.

Section 2  
    spark-submit --py-files s3://iteso-bigdata-alicortes/src/spark_utils.py \
        s3://iteso-bigdata-alicortes/src/project.py \
        --input  s3://iteso-bigdata-alicortes/data/project/ \
        --report s3://iteso-bigdata-alicortes/debug/report_transformations.txt

Section 3  
        --report s3://iteso-bigdata-alicortes/debug/report_persistence.txt \
        --output s3://iteso-bigdata-alicortes/output/project/
"""
import argparse

from spark_utils import SparkUtils


def parse_args():
    parser = argparse.ArgumentParser(description="Overture buildings batch pipeline")
    parser.add_argument("--input", required=True, help="Folder with type=building Parquet files")
    parser.add_argument("--report", required=True, help="Text report to write (s3://... or local path)")
    parser.add_argument("--output", help="Output folder. when given, the result is persisted as Parquet")
    parser.add_argument("--rows", type=int, default=5, help="Number of sample records ")
    parser.add_argument("--count", action="store_true", help="Also count raw and final rows (full scans)")
    parser.add_argument("--sample", type=float, help="Fraction of buildings to process (quick tests)")
    parser.add_argument("--skip-nulls", action="store_true", help="Skip the %% of nulls in the report (saves a full scan of the raw data)")
    return parser.parse_args()


def main():
    args = parse_args()
    su = SparkUtils("overture-buildings-pipeline")

    try: 
        df_raw = su.read_buildings(args.input, args.sample)

        # Transformations  
        df_clean = su.clean_buildings(df_raw)
        df_derived = su.add_derived_columns(df_clean)
        df_final = su.select_output_columns(su.filter_relevant(df_derived))
        # Cached so the dedup shuffle runs once and is reused by the write and every report action
        df_final = su.persist(df_final)

        # Persistence 
        if args.output:
            su.write_partitioned(df_final, args.output, ["region", "housing_type"])

        df_sorted = su.sort_by_size(df_final)

        # Actions for the report
        report = su.pipeline_report(df_raw, df_final, df_sorted,
                                    title=f"Input: {args.input}\nOutput: {args.output}",
                                    rows=args.rows, count_rows=args.count,
                                    null_stats=not args.skip_nulls)
    except Exception as e:
        # Still leave a report behind, so a failed step can be diagnosed from S3
        su.write_text(args.report, f"Input: {args.input}\n\n=== ERROR ===\n{type(e).__name__}: {e}\n")
        su.spark.stop()
        raise

    su.write_text(args.report, report)
    print(f"Debug report written to {args.report}")
    su.spark.stop()


if __name__ == "__main__":
    main()
