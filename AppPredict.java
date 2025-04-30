package com.winequality;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.col;

public class AppPredict {


    public static Dataset<Row> cleanData(Dataset<Row> df) {
        for (String colName : df.columns()) {
            String cleanColName = colName.replace("\"", "");
            df = df.withColumn(cleanColName, col(colName).cast(DataTypes.DoubleType));
            if (!cleanColName.equals(colName)) {
                df = df.drop(colName);
            }
        }
        return df;
    }

    public static void main(String[] args) {
        // Initialize Spark
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Prediction")
                .master("local[*]")  // local mode for testing
                .getOrCreate();

        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("ERROR");

        
        String testPath = "s3a://wine-train-dataset-pa2/ValidationDataset.csv"; 


        Dataset<Row> testData = spark.read()
                .format("csv")
                .option("header", "true")
                .option("sep", ";")
                .option("inferSchema", "true")
                .load(testPath);

        testData = cleanData(testData);

        String[] modelNames = {
                "LogisticRegression",
                "RandomForestClassifier",
                "DecisionTreeClassifier",
		"NaiveBayes",
                "MultilayerPerceptronClassifier"
        };

        for (String modelName : modelNames) {
            System.out.println("\nEvaluating model: " + modelName);
            String modelPath = "s3a://wine-train-dataset-pa2/" + modelName;

            try {
                PipelineModel model = PipelineModel.load(modelPath);

                Dataset<Row> predictions = model.transform(testData);

                MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                        .setLabelCol("label")            
                        .setPredictionCol("prediction");

                double accuracy = evaluator.setMetricName("accuracy").evaluate(predictions);
                double f1 = evaluator.setMetricName("f1").evaluate(predictions);
                double precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions);
                double recall = evaluator.setMetricName("weightedRecall").evaluate(predictions);

                System.out.printf("Model: %s | Accuracy: %.4f | F1 Score: %.4f | Precision: %.4f | Recall: %.4f%n",
                        modelName, accuracy, f1, precision, recall);
            } catch (Exception e) {
                System.err.printf("Error evaluating model %s: %s%n", modelName, e.getMessage());
            }
        }

        spark.stop();
    }
}

