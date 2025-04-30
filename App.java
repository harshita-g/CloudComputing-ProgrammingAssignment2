package com.winequality;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.apache.spark.sql.functions.col;

public class App {
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
        SparkSession spark = SparkSession.builder()
                .appName("wine_quality_prediction")
                .master("local[*]")
                .getOrCreate();

        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("ERROR");

        String inputPath = "s3a://wine-train-dataset-pa2/TrainingDataset.csv";
        String validPath = "s3a://wine-train-dataset-pa2/ValidationDataset.csv";

        Dataset<Row> trainData = spark.read()
                .format("csv")
                .option("header", "true")
                .option("sep", ";")
                .option("inferschema", "true")
                .load(inputPath);
        trainData = cleanData(trainData);

        Dataset<Row> validData = spark.read()
                .format("csv")
                .option("header", "true")
                .option("sep", ";")
                .option("inferschema", "true")
                .load(validPath);
        validData = cleanData(validData);

        String[] featureColumns = new String[]{
                "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                "pH", "sulphates", "alcohol"
        };

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureColumns)
                .setOutputCol("scaledFeatures");


        StringIndexer indexer = new StringIndexer()
                .setInputCol("quality")
                .setOutputCol("label")
                .setHandleInvalid("skip");

        List<Classifier> classifiers = new ArrayList<>();
        classifiers.add(new LogisticRegression()
                .setLabelCol("label")
                .setFeaturesCol("scaledFeatures")
                .setMaxIter(100)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .setFamily("multinomial"));
        classifiers.add(new RandomForestClassifier()
                .setLabelCol("label")
                .setFeaturesCol("scaledFeatures")
                .setNumTrees(50));
        classifiers.add(new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("scaledFeatures"));
        classifiers.add(new NaiveBayes()
                .setLabelCol("label")
                .setFeaturesCol("scaledFeatures"));
        classifiers.add(new MultilayerPerceptronClassifier()
                .setLabelCol("label")
                .setFeaturesCol("scaledFeatures")
                .setLayers(new int[]{11, 62, 32, 16, 10}));

        for (Classifier classifier : classifiers) {
            System.out.println("Training and evaluating: " + classifier.getClass().getSimpleName());

            Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
                    assembler, indexer, classifier
            });

            PipelineModel model = pipeline.fit(trainData);
            Dataset<Row> predictions = model.transform(validData);

            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("label")
                    .setPredictionCol("prediction");

            double accuracy = evaluator.setMetricName("accuracy").evaluate(predictions);
            double f1Score = evaluator.setMetricName("f1").evaluate(predictions);
            double precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions);
            double recall = evaluator.setMetricName("weightedRecall").evaluate(predictions);

            System.out.printf("Model: %s | Accuracy: %.4f | F1 Score: %.4f | Precision: %.4f | Recall: %.4f\n",
                    classifier.getClass().getSimpleName(), accuracy, f1Score, precision, recall);

            String modelPath = "s3a://wine-train-dataset-pa2/" + classifier.getClass().getSimpleName();
            try {
                model.write().overwrite().save(modelPath);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            System.out.println("Model saved to: " + modelPath);
        }

        spark.stop();
    }
}

