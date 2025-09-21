using System;
using System.Collections.Generic;
using System.Linq;

public class GameReviewClassifier
{
    // A class to store the review data for each game.
    public class GameReview
    {
        public string Name { get; set; }
        public double ReviewScore { get; set; }
        public string ReviewType { get; set; } // "Positive" or "Negative"
    }

    // A class to represent a single data point for our models,
    // containing a single feature (the score) and a numerical label.
    public class DataPoint
    {
        public double Feature { get; set; }
        public int Label { get; set; } // 1 for Positive, 0 for Negative
    }

    // A class for Naive Bayes to hold the binned score and label.
    public class BinnedDataPoint
    {
        public string BinnedScore { get; set; }
        public int Label { get; set; }
    }

    public static void Main(string[] args)
    {
        Console.WriteLine("Welcome to the Game Review Classifier!");
        Console.WriteLine("This program will classify game reviews using a numerical score (0-10) with three machine learning models built from scratch.");

        // Start with an empty list of game reviews.
        var gameReviews = new List<GameReview>();

        // Get user input for review games data.
        Console.WriteLine("\n--- Reviewed Games Input ---");
        int numGamesToPredict = 0;
        bool isValidInput = false;
        while (!isValidInput)
        {
            Console.Write("Enter the number of games to enter: ");
            string input = Console.ReadLine();
            if (int.TryParse(input, out numGamesToPredict) && numGamesToPredict > 0)
            {
                isValidInput = true;
            }
            else
            {
                Console.WriteLine("Invalid input. Please enter a positive integer.");
            }
        }

        for (int i = 0; i < numGamesToPredict; i++)
        {
            Console.WriteLine($"\n--- Entering Game {i + 1} of {numGamesToPredict}");
            Console.Write("Enter the game name: ");
            string name = Console.ReadLine();
            double score = -1;
            bool isValidScore = false;
            while (!isValidScore)
            {
                Console.Write("Enter overall review score (0.0-10.0): ");
                string scoreInput = Console.ReadLine();
                if (double.TryParse(scoreInput, out score) && score >= 0.0 && score <= 10.0)
                {
                    isValidScore = true;
                }
                else
                {
                    Console.WriteLine("Invalid score. Please enter a number between 0.0 and 10.0.");
                }
            }
            string type = "";
            bool isValidType = false;
            while (!isValidType)
            {
                Console.Write("Enter review type (Positive/Negative): ");
                type = Console.ReadLine();
                if (type.Equals("Positive", StringComparison.OrdinalIgnoreCase) || type.Equals("Negative", StringComparison.OrdinalIgnoreCase))
                {
                    isValidType = true;
                }
                else
                {
                    Console.WriteLine("Invalid type. Please enter 'Positive' or 'Negative'.");
                }
            }
            gameReviews.Add(new GameReview { Name = name, ReviewScore = score, ReviewType = type });
        }

        // --- Display Data Table ---
        Console.WriteLine("\n--- Data Table ---");
        PrintDataTable(gameReviews);

        // --- Data Preprocessing ---
        var preprocessedData = PreprocessData(gameReviews);

        // Splitting data: The last game entered will be the test data. All others are training data.
        var trainingData = preprocessedData.Take(preprocessedData.Count - 1).ToList();
        var testData = preprocessedData.Skip(preprocessedData.Count - 1).ToList();

        // --- Regression Model Training and Evaluation ---
        Console.WriteLine("\n--- Training Regression Model ---");
        var regressionModel = new RegressionModel();
        regressionModel.Train(trainingData);
        Console.WriteLine("Regression Model Predictions:");
        double regressionAccuracy = EvaluateModel(regressionModel, testData);
        Console.WriteLine($"\nRegression Model Accuracy: {regressionAccuracy:P2}");

        // --- Naive Bayes Classifier Training and Evaluation ---
        Console.WriteLine("\n--- Training Naive Bayes Classifier ---");
        var binnedData = BinData(gameReviews);
        var binnedTrainingData = binnedData.Take(binnedData.Count - 1).ToList();
        var binnedTestData = binnedData.Skip(binnedData.Count - 1).ToList();
        var naiveBayesModel = new NaiveBayesClassifier();
        naiveBayesModel.Train(binnedTrainingData);
        Console.WriteLine("Naive Bayes Predictions:");
        double naiveBayesAccuracy = EvaluateModel(naiveBayesModel, binnedTestData);
        Console.WriteLine($"\nNaive Bayes Accuracy: {naiveBayesAccuracy:P2}");

        // --- Multi-layer Perceptron (MLP) Training and Evaluation ---
        Console.WriteLine("\n--- Training Multi-layer Perceptron (MLP) ---");
        var mlpModel = new MLP();
        mlpModel.Train(trainingData);
        Console.WriteLine("MLP Predictions:");
        double mlpAccuracy = EvaluateModel(mlpModel, testData);
        Console.WriteLine($"\nMLP Accuracy: {mlpAccuracy:P2}");

        // --- New Game Prediction ---
        Console.WriteLine("\n--- New Game Prediction ---");
        Console.WriteLine("Let's predict the review type for a new, incoming game.");
        Console.Write("Enter the name of the new game: ");
        string newGameName = Console.ReadLine();
        double newGameScore = -1;
        bool isValidNewScore = false;
        while (!isValidNewScore)
        {
            Console.Write("Enter the review score for the new game (0.0-10.0): ");
            string newScoreInput = Console.ReadLine();
            if (double.TryParse(newScoreInput, out newGameScore) && newGameScore >= 0.0 && newGameScore <= 10.0)
            {
                isValidNewScore = true;
            }
            else
            {
                Console.WriteLine("Invalid score. Please enter a number between 0.0 and 10.0.");
            }
        }

        // Regression Prediction
        double regRawOutput = regressionModel.GetPredictionOutput(newGameScore);
        int regPredictionLabel = (regRawOutput >= 0.5) ? 1 : 0;
        string regPredictionType = (regPredictionLabel == 1) ? "Positive" : "Negative";
        
        // Calculate probability using sigmoid function for a more meaningful value
        double probability = 1.0 / (1.0 + Math.Exp(-regRawOutput));

        Console.WriteLine($"\nRegression Model predicts '{newGameName}' is a {regPredictionType} review.");
        Console.WriteLine($"\tRaw Output: {regRawOutput:F4}");
        Console.WriteLine($"\tProbability: {probability:P2}");

        // Naive Bayes Prediction
        string binnedNewScore;
        if (newGameScore < 5.0)
        {
            binnedNewScore = "low";
        }
        else if (newGameScore < 8.0)
        {
            binnedNewScore = "medium";
        }
        else
        {
            binnedNewScore = "high";
        }
        int nbPredictionLabel = naiveBayesModel.Predict(binnedNewScore);
        string nbPredictionType = (nbPredictionLabel == 1) ? "Positive" : "Negative";
        Console.WriteLine($"Naive Bayes Classifier predicts '{newGameName}' is a {nbPredictionType} review.");
        
        Console.ReadLine();
    }

    public static void PrintDataTable(List<GameReview> reviews)
    {
        Console.WriteLine(new string('-', 60));
        Console.WriteLine($"| {"Game Name",-20} | {"Review Score",-15} | {"Review Type",-10} |");
        Console.WriteLine(new string('-', 60));
        foreach (var review in reviews)
        {
            Console.WriteLine($"| {review.Name,-20} | {review.ReviewScore,-15:F1} | {review.ReviewType,-10} |");
        }
        Console.WriteLine(new string('-', 60));
    }

    // This method converts a list of GameReview objects into DataPoint objects for numerical models.
    private static List<DataPoint> PreprocessData(List<GameReview> reviews)
    {
        Console.WriteLine("\n--- Preprocessing Data ---");
        var preprocessed = new List<DataPoint>();
        foreach (var review in reviews)
        {
            int label = review.ReviewType.Equals("Positive", StringComparison.OrdinalIgnoreCase) ? 1 : 0;
            preprocessed.Add(new DataPoint { Feature = review.ReviewScore, Label = label });
        }
        Console.WriteLine("Data converted to numerical data points.");
        return preprocessed;
    }

    // This method bins the numerical data into categories for the Naive Bayes Classifier.
    private static List<BinnedDataPoint> BinData(List<GameReview> reviews)
    {
        Console.WriteLine("--- Binning Data for Naive Bayes ---");
        var binnedData = new List<BinnedDataPoint>();
        foreach (var review in reviews)
        {
            string binnedScore;
            if (review.ReviewScore < 5.0)
            {
                binnedScore = "low";
            }
            else if (review.ReviewScore < 8.0)
            {
                binnedScore = "medium";
            }
            else
            {
                binnedScore = "high";
            }

            int label = review.ReviewType.Equals("Positive", StringComparison.OrdinalIgnoreCase) ? 1 : 0;
            binnedData.Add(new BinnedDataPoint { BinnedScore = binnedScore, Label = label });
        }
        return binnedData;
    }

    // Evaluates a model's accuracy on a given test set.
    private static double EvaluateModel(object model, object testData)
    {
        if (testData is List<DataPoint> numericTestData)
        {
            if (numericTestData.Count == 0) return 0;
            int correct = 0;
            for (int i = 0; i < numericTestData.Count; i++)
            {
                int predictedLabel = -1;
                if (model is RegressionModel regModel)
                {
                    predictedLabel = regModel.Predict(numericTestData[i].Feature);
                }
                else if (model is MLP mlpModel)
                {
                    predictedLabel = mlpModel.Predict(numericTestData[i].Feature);
                }
                string actualType = (numericTestData[i].Label == 1) ? "Positive" : "Negative";
                string predictedType = (predictedLabel == 1) ? "Positive" : "Negative";
                Console.WriteLine($"\tActual: {actualType,-10} | Predicted: {predictedType,-10} -> {(predictedLabel == numericTestData[i].Label ? "Correct" : "Incorrect")}");
                if (predictedLabel == numericTestData[i].Label)
                {
                    correct++;
                }
            }
            return (double)correct / numericTestData.Count;
        }
        else if (testData is List<BinnedDataPoint> binnedTestData)
        {
            if (binnedTestData.Count == 0) return 0;
            int correct = 0;
            for (int i = 0; i < binnedTestData.Count; i++)
            {
                int predictedLabel = -1;
                if (model is NaiveBayesClassifier nbModel)
                {
                    predictedLabel = nbModel.Predict(binnedTestData[i].BinnedScore);
                }
                string actualType = (binnedTestData[i].Label == 1) ? "Positive" : "Negative";
                string predictedType = (predictedLabel == 1) ? "Positive" : "Negative";
                Console.WriteLine($"\tActual: {actualType,-10} | Predicted: {predictedType,-10} -> {(predictedLabel == binnedTestData[i].Label ? "Correct" : "Incorrect")}");
                if (predictedLabel == binnedTestData[i].Label)
                {
                    correct++;
                }
            }
            return (double)correct / binnedTestData.Count;
        }
        return 0;
    }

    // --- REGRESSION MODEL (Simplified for Classification) ---
    // A simplified Linear Regression model that uses gradient descent to learn a weight and a bias.
    // The final prediction is a binary classification based on a threshold.
    public class RegressionModel
    {
        private double weight;
        private double bias;
        private double learningRate = 0.001;
        private int epochs = 20000;

        public void Train(List<DataPoint> data)
        {
            if (data.Count == 0) return;
            weight = 0;
            bias = 0;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var point in data)
                {
                    double prediction = weight * point.Feature + bias;
                    double error = point.Label - prediction;

                    weight += learningRate * error * point.Feature;
                    bias += learningRate * error;
                }
            }
            Console.WriteLine("Training complete.");
        }
        
        // This method returns the binary prediction (1 or 0).
        public int Predict(double feature)
        {
            double prediction = weight * feature + bias;
            return prediction >= 0.5 ? 1 : 0;
        }

        // This new method returns the raw, un-thresholded output.
        public double GetPredictionOutput(double feature)
        {
            return weight * feature + bias;
        }
    }

    // --- NAIVE BAYES CLASSIFIER ---
    // This classifier uses Bayes' theorem on binned (categorized) data.
    public class NaiveBayesClassifier
    {
        private Dictionary<string, double> positiveProbabilities;
        private Dictionary<string, double> negativeProbabilities;
        private double positiveClassProb;
        private double negativeClassProb;
        private HashSet<string> allBins;

        public void Train(List<BinnedDataPoint> data)
        {
            if (data.Count == 0) return;

            var positiveData = data.Where(d => d.Label == 1).ToList();
            var negativeData = data.Where(d => d.Label == 0).ToList();

            positiveClassProb = (double)positiveData.Count / data.Count;
            negativeClassProb = (double)negativeData.Count / data.Count;

            allBins = new HashSet<string>(data.Select(d => d.BinnedScore));

            positiveProbabilities = GetFeatureProbabilities(positiveData);
            negativeProbabilities = GetFeatureProbabilities(negativeData);

            Console.WriteLine("Training complete.");
        }

        private Dictionary<string, double> GetFeatureProbabilities(List<BinnedDataPoint> data)
        {
            var probabilities = new Dictionary<string, double>();
            var binCounts = new Dictionary<string, int>();

            foreach (var bin in allBins)
            {
                binCounts[bin] = 0;
            }

            foreach (var point in data)
            {
                if (binCounts.ContainsKey(point.BinnedScore))
                {
                    binCounts[point.BinnedScore]++;
                }
            }

            int totalFeatures = data.Count;
            int numBins = allBins.Count;

            foreach (var bin in allBins)
            {
                // Laplace smoothing to prevent zero probabilities
                probabilities[bin] = (double)(binCounts[bin] + 1) / (totalFeatures + numBins);
            }
            return probabilities;
        }

        public int Predict(string binnedFeature)
        {
            double positiveScore = Math.Log(positiveClassProb);
            double negativeScore = Math.Log(negativeClassProb);

            if (positiveProbabilities.ContainsKey(binnedFeature))
            {
                positiveScore += Math.Log(positiveProbabilities[binnedFeature]);
            }
            else
            {
                positiveScore += Math.Log(1.0 / (positiveProbabilities.Count + allBins.Count));
            }

            if (negativeProbabilities.ContainsKey(binnedFeature))
            {
                negativeScore += Math.Log(negativeProbabilities[binnedFeature]);
            }
            else
            {
                negativeScore += Math.Log(1.0 / (negativeProbabilities.Count + allBins.Count));
            }

            return positiveScore > negativeScore ? 1 : 0;
        }
    }

    // --- MULTI-LAYER PERCEPTRON (MLP) ---
    // A simple neural network with one hidden layer, using backpropagation for training.
    public class MLP
    {
        private double[] hiddenWeights;
        private double hiddenBias;
        private double outputWeight;
        private double outputBias;
        private double learningRate = 0.01;
        private int epochs = 20000;
        private int hiddenLayerSize = 5;

        public void Train(List<DataPoint> data)
        {
            if (data.Count == 0) return;
            Random rand = new Random();

            hiddenWeights = new double[hiddenLayerSize];
            for (int i = 0; i < hiddenLayerSize; i++)
            {
                hiddenWeights[i] = rand.NextDouble() * 0.1;
            }
            hiddenBias = rand.NextDouble() * 0.1;
            outputWeight = rand.NextDouble() * 0.1;
            outputBias = rand.NextDouble() * 0.1;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var point in data)
                {
                    // Forward Propagation
                    double[] hiddenOutput = new double[hiddenLayerSize];
                    for (int i = 0; i < hiddenLayerSize; i++)
                    {
                        hiddenOutput[i] = Sigmoid(point.Feature * hiddenWeights[i] + hiddenBias);
                    }
                    double finalOutput = Sigmoid(DotProduct(hiddenOutput) + outputBias);

                    // Backpropagation
                    double outputDelta = finalOutput - point.Label;
                    double[] hiddenDelta = new double[hiddenLayerSize];
                    for (int i = 0; i < hiddenLayerSize; i++)
                    {
                        hiddenDelta[i] = outputDelta * outputWeight * SigmoidDerivative(hiddenOutput[i]);
                    }

                    // Update weights and biases
                    outputWeight -= learningRate * outputDelta * DotProduct(hiddenOutput);
                    outputBias -= learningRate * outputDelta;
                    for (int i = 0; i < hiddenLayerSize; i++)
                    {
                        hiddenWeights[i] -= learningRate * hiddenDelta[i] * point.Feature;
                        hiddenBias -= learningRate * hiddenDelta[i];
                    }
                }
            }
            Console.WriteLine("Training complete.");
        }

        public int Predict(double feature)
        {
            double[] hiddenOutput = new double[hiddenLayerSize];
            for (int i = 0; i < hiddenLayerSize; i++)
            {
                hiddenOutput[i] = Sigmoid(feature * hiddenWeights[i] + hiddenBias);
            }
            double finalOutput = Sigmoid(DotProduct(hiddenOutput) + outputBias);
            return finalOutput >= 0.5 ? 1 : 0;
        }

        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        private double SigmoidDerivative(double x) => x * (1 - x);
        private double DotProduct(double[] a)
        {
            double sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                sum += a[i] * outputWeight;
            }
            return sum;
        }
    }
}