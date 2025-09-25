from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, lit
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def initialize_spark():
    spark = SparkSession.builder.appName("Contcrisis Project").getOrCreate()
    return spark

def load_dataset(path, spark):
    df = spark.read.csv(path, header=True, inferSchema=True)
    return df

def get_dataset_dimensions(df):
    num_rows = df.count()
    num_cols = len(df.columns)
    return num_rows, num_cols

def handle_nulls(df):

    numeric_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ["double", "int", "float", "long"]]
    categorical_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() == "string"]
    for col_name in numeric_cols:
        mean_value = df.select(mean(col(col_name))).first()[0]
        if mean_value is not None:
            df = df.withColumn(col_name, when(col(col_name).isNull(), mean_value).otherwise(col(col_name)))
    for col_name in categorical_cols:
        df = df.withColumn(col_name, when(col(col_name).isNull(), lit("missing")).otherwise(col(col_name)))

    return df

def define_target_variable(df, target_col):
    indexer = StringIndexer(inputCol=target_col, outputCol="label", handleInvalid="keep")
    df = indexer.fit(df).transform(df)
    df = df.withColumn("label", col("label").cast("integer"))
    return df

def transform_features(df):
    ignore_cols = ["label", "weight"]

    categorical_cols = [
        field.name for field in df.schema.fields
        if isinstance(field.dataType, StringType) and field.name not in ignore_cols
    ]

    numeric_cols = [
        field.name for field in df.schema.fields
        if not isinstance(field.dataType, StringType) and field.name not in ignore_cols
    ]

    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="keep")
        for col in categorical_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded")
        for col in categorical_cols
    ]

    encoded_cols = [f"{col}_encoded" for col in categorical_cols]
    final_features = numeric_cols + encoded_cols

    assembler = VectorAssembler(inputCols=final_features, outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [assembler])
    model = pipeline.fit(df)
    df_transformed = model.transform(df)

    return df_transformed


def compute_correlations(df, target_col):

    numeric_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ["double", "int", "float", "long"]]
    numeric_cols = [col for col in numeric_cols if col != target_col]

    correlations = {}
    for col_name in numeric_cols:
        corr_value = df.stat.corr(col_name, target_col)
        correlations[col_name] = corr_value

    sorted_correlations = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))

    return sorted_correlations


def analyze_class_imbalance(df, label_col="label"):
    return df.groupBy(label_col).count().orderBy("count", ascending=False)

def add_class_weights(df, label_col="label", weight_col="weight"):
    class_counts = df.groupBy(label_col).count().collect()
    counts_dict = {row[label_col]: row["count"] for row in class_counts}

    total = sum(counts_dict.values())
    class_weights = {k: total / v for k, v in counts_dict.items()}

    weight_expr = when(col(label_col) == lit(0), lit(class_weights[0])).otherwise(lit(class_weights[1]))

    df = df.withColumn(weight_col, weight_expr)
    return df

def split_dataset(df, test_size=0.2, seed=42):
    train_df, test_df = df.randomSplit([1 - test_size, test_size], seed=seed)
    return train_df, test_df

def train_model(model, train_df, label_col="label", features_col="features", weight_col="weight"):
    model = model.setLabelCol(label_col).setFeaturesCol(features_col).setWeightCol(weight_col)
    trained_model = model.fit(train_df)
    return trained_model

def model_evaluator(model, test_df):
    full_pred = model.transform(test_df)
    predictions = full_pred.select("label", "prediction")

    evaluators = {
        "accuracy": MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy"),
        "f1": MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1"),
        "auc": BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    }

    metrics = {}
    tp = predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
    fp = predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
    fn = predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()

    metrics["precision"] = tp / (tp + fp) if (tp + fp) else 0
    metrics["recall"] = tp / (tp + fn) if (tp + fn) else 0

    for name, evaluator in evaluators.items():
        metrics[name] = evaluator.evaluate(full_pred)

    return metrics

def compute_mcc(predictions_df):

    metrics = predictions_df.select(
        when((col("label") == 1) & (col("prediction") == 1), 1).otherwise(0).alias("TP"),
        when((col("label") == 0) & (col("prediction") == 1), 1).otherwise(0).alias("FP"),
        when((col("label") == 1) & (col("prediction") == 0), 1).otherwise(0).alias("FN"),
        when((col("label") == 0) & (col("prediction") == 0), 1).otherwise(0).alias("TN")
    ).groupBy().sum().collect()[0]

    tp = metrics["sum(TP)"]
    fp = metrics["sum(FP)"]
    fn = metrics["sum(FN)"]
    tn = metrics["sum(TN)"]

    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    mcc = numerator / denominator if denominator != 0 else 0.0

    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    return mcc


def print_model_metrics(model_name, metrics_dict):
    print(f"\n{model_name}:")
    for k, v in metrics_dict.items():
        print(f"{k.capitalize()}: {v:.4f}")


def plot_roc_curve(model, test_df, model_name = "modello"):
    predictions = model.transform(test_df).select("label", "probability")

    preds = predictions.rdd.map(lambda row: (row["label"], float(row["probability"][1]))).collect()

    y_true = [int(label) for label, prob in preds]
    y_scores = [prob for label, prob in preds]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC curve of {model_name}")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():

    path = "finaldataset_1.csv"
    target_col = "contcrisis"

    spark = initialize_spark()

    df = load_dataset(path, spark)

    num_rows, num_cols = get_dataset_dimensions(df)
    print(f"Numero di osservazioni: {num_rows}")
    print(f"Numero di features: {num_cols}")

    df = handle_nulls(df)
    df = define_target_variable(df, target_col)

    class_distribution = analyze_class_imbalance(df)
    print("Distribuzione della variabile target (label):")
    class_distribution.show()

    df = add_class_weights(df)
    df = transform_features(df)

    correlations = compute_correlations(df, target_col)
    print("\nCorrelazioni con il target:")
    for k, v in correlations.items():
        print(f"{k}: {v:.4f}")

    train_df, test_df = split_dataset(df)
    print(f"Training set: {train_df.count()} righe")
    print(f"Test set: {test_df.count()} righe")

    rf_model = RandomForestClassifier(numTrees=100, maxDepth=5)
    trained_rf = train_model(rf_model, train_df)
    rf_predictions = trained_rf.transform(test_df)
    print("random forest addestrato")
    metrics_rf = model_evaluator(trained_rf, test_df)
    compute_mcc(rf_predictions)
    print_model_metrics("Random Forest", metrics_rf)

    lr_model = LogisticRegression(maxIter=100, regParam=0.01)
    trained_lr = train_model(lr_model, train_df)
    lr_predictions = trained_lr.transform(test_df)
    print("logistic regression addestrato")
    metrics_lr = model_evaluator(trained_lr, test_df)
    compute_mcc(lr_predictions)
    print_model_metrics("Logistic Regression", metrics_lr)

    plot_roc_curve(trained_rf, test_df, model_name="Random Forest")
    plot_roc_curve(trained_lr, test_df, model_name="Logistic Regression")

if __name__ == "__main__":
    main()
