using System.Diagnostics;
using System.Globalization;
using Italbytz.Adapters.Algorithms.AI.Search.GP;
using Italbytz.AI.Util;
using Italbytz.ML;
using logicGP.Tests.Data.Real;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace logicGP.Tests;

public abstract class RealTests
{
    protected int[]
        Seeds =
        [
            42, 7, 13, 23, 3, 666, 777, 9, 17, 21
        ]; //[42, 23, 7, 3, 99, 1, 0, 8, 15, 16];

    protected string? LogFile { get; set; } = null;
    protected StreamWriter? LogWriter { get; set; }
    protected StreamWriter? ResultWriter { get; set; }

    private ITransformer Train<TLabel>(MLContext mlContext,
        IEstimator<ITransformer> generalTrainer, IDataView trainData,
        LookupMap<TLabel>[] lookupData, int generations = 100)
    {
        var lookupIdvMap = mlContext.Data.LoadFromEnumerable(lookupData);
        if (generalTrainer is not LogicGpTrainerBase<ITransformer> trainer)
            throw new ArgumentException(
                "The trainer must be of type LogicGpTrainerBase<ITransformer>");
        trainer.LabelKeyToValueDictionary =
            LookupMap<TLabel>.KeyToValueMap(lookupData);
        trainer.Label = "Label";
        trainer.MaxGenerations = generations;
        var pipeline = GetPipeline(trainer, lookupIdvMap);
        return pipeline.Fit(trainData);
    }

    
    protected void SaveCvSplit(
        IDataView data, string fileName)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var cvResults = mlContext.Data.CrossValidationSplit(data);
        var foldIndex = 0;

        foreach (var fold in cvResults)
        {
            var trainSet = fold.TrainSet;
            var testSet = fold.TestSet;
            var dataFolder =
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data");
            if (!Directory.Exists(dataFolder))
                Directory.CreateDirectory(dataFolder);

            trainSet.SaveAsCsv(Path.Combine(dataFolder,
                $"{fileName}_fold_{foldIndex}_train.csv"));
            testSet.SaveAsCsv(Path.Combine(dataFolder,
                $"{fileName}_fold_{foldIndex}_test.csv"));
            foldIndex++;
        }
    }

    protected LogicGpFlrwMacroMulticlassTrainer GetFlRwMacroTrainer(int classes)
    {
        var services = new ServiceCollection().AddServices();
        var serviceProvider = services.BuildServiceProvider();
        var trainer =
            serviceProvider
                .GetRequiredService<LogicGpFlrwMacroMulticlassTrainer>();
        trainer.Classes = classes;
        return trainer;
    }

    protected LogicGpFlrwMicroMulticlassTrainer GetFlRwMicroTrainer(int classes)
    {
        var services = new ServiceCollection().AddServices();
        var serviceProvider = services.BuildServiceProvider();
        var trainer =
            serviceProvider
                .GetRequiredService<LogicGpFlrwMicroMulticlassTrainer>();
        trainer.Classes = classes;
        return trainer;
    }

    protected IDataView TestFlRw<TLabel>(
        IEstimator<ITransformer> generalTrainer,
        IDataView trainData, IDataView testData,
        LookupMap<TLabel>[] lookupData, int generations = 100)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var mlModel = Train(mlContext, generalTrainer, trainData, lookupData,
            generations);
        Assert.IsNotNull(mlModel);
        return mlModel.Transform(testData);
    }

    public void SimulateFlRw<TLabel>(IEstimator<ITransformer> trainer,
        IDataView data, LookupMap<TLabel>[] lookupData, int generations = 10000)
    {
        var logFolder =
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "logs");
        if (!Directory.Exists(logFolder))
            Directory.CreateDirectory(logFolder);
        var timeStamp = DateTime.Now.ToString("yyyyMMddHHmmss");
        if (LogFile != null)
        {
            var logPath = Path.Combine(logFolder,
                $"{LogFile}_{timeStamp}.log");
            LogWriter = new StreamWriter(logPath);
            var resultPath = Path.Combine(logFolder,
                $"{LogFile}_{timeStamp}.csv");
            ResultWriter = new StreamWriter(resultPath);
        }

        foreach (var mlSeed in Seeds)
        foreach (var randomSeed in Seeds)
        {
            var mlContext = new MLContext(mlSeed);
            ThreadSafeRandomNetCore.Seed = randomSeed;
            var trainTestSplit = mlContext.Data.TrainTestSplit(data, 0.2);

            var trainData = trainTestSplit.TrainSet;
            var testData = trainTestSplit.TestSet;

            var mlModel = Train(mlContext, trainer, trainData, lookupData,
                generations);
            Assert.IsNotNull(mlModel);
            var chosenIndividual =
                ((LogicGpTrainerBase<ITransformer>)trainer)
                .ChosenIndividual;
            var testResults = mlModel.Transform(testData);
            LogWriter?.WriteLine(
                $"ML seed: {mlSeed}, Random seed: {randomSeed}, Generations: {generations}");
            LogWriter?.WriteLine();
            LogWriter?.Write(chosenIndividual.ToString());
            // ToDO: lookupData größe 2 -> Binär
            LogWriter?.WriteLine();
            if (lookupData.Length > 2)
            {
                var metrics = new MLContext().MulticlassClassification
                    .Evaluate(testResults);
                LogWriter?.WriteLine(
                    $"MacroAccuracy: {metrics.MacroAccuracy.ToString(CultureInfo.InvariantCulture)}");
                ResultWriter?.WriteLine(
                    metrics.MacroAccuracy.ToString(CultureInfo
                        .InvariantCulture));
            }
            else
            {
                var metrics = new MLContext().BinaryClassification
                    .Evaluate(testResults);
                LogWriter?.WriteLine(
                    $"Accuracy: {metrics.Accuracy.ToString(CultureInfo.InvariantCulture)}");
                LogWriter?.WriteLine(
                    $"AUC: {metrics.AreaUnderRocCurve.ToString(CultureInfo.InvariantCulture)}");
                LogWriter?.WriteLine(
                    $"F1Score: {metrics.F1Score.ToString(CultureInfo.InvariantCulture)}");
                LogWriter?.WriteLine(
                    $"AUPRC: {metrics.AreaUnderPrecisionRecallCurve.ToString(CultureInfo.InvariantCulture)}");
                ResultWriter?.WriteLine(
                    $"{metrics.Accuracy.ToString(CultureInfo.InvariantCulture)},{metrics.AreaUnderRocCurve.ToString(CultureInfo.InvariantCulture)},{metrics.F1Score.ToString(CultureInfo.InvariantCulture)},{metrics.F1Score.ToString(CultureInfo.InvariantCulture)},{metrics.AreaUnderPrecisionRecallCurve.ToString(CultureInfo.InvariantCulture)}"
                );
            }

            LogWriter?.Flush();
            ResultWriter?.Flush();
            /*ResultWriter?.WriteLine(
                    chosenIndividual.LatestKnownFitness[0].ToString(
                        CultureInfo.InvariantCulture));*/
        }

        LogWriter?.Close();
        ResultWriter?.Close();
    }

    protected abstract IEstimator<ITransformer?> GetPipeline(
        LogicGpTrainerBase<ITransformer> trainer, IDataView lookupIdvMap);


    private void CleanUp()
    {
        var directory = AppDomain.CurrentDomain.BaseDirectory;
        var projectFiles = Directory.GetFiles(directory, "*.cs");
        foreach (var file in projectFiles) File.Delete(file);
        projectFiles = Directory.GetFiles(directory, "*.csproj");
        foreach (var file in projectFiles) File.Delete(file);
    }


    private void RunAutoMLForConfig(string modelFileName)
    {
        var timeStamp = DateTime.Now.ToString("yyyyMMddHHmmss");
        var mlnet = new Process();
        mlnet.StartInfo.FileName = "mlnet";
        mlnet.StartInfo.WorkingDirectory =
            AppDomain.CurrentDomain.BaseDirectory;
        //mlnet.StartInfo.Arguments = $"train --training-config config.mbconfig --log-file-path {timeStamp}.log";
        //mlnet.StartInfo.Arguments = "train --training-config config.mbconfig";
        mlnet.StartInfo.Arguments =
            $"train --training-config {modelFileName}.mbconfig -v q";
        mlnet.Start();
        mlnet.WaitForExit();
    }

    private void PrintAccuracies(Dictionary<string, float> bestMacroaccuracy)
    {
        foreach (var entry in bestMacroaccuracy)
            Console.WriteLine(
                $"{entry.Key}: {entry.Value.ToString(CultureInfo.InvariantCulture)}");

        Console.WriteLine(
            (bestMacroaccuracy["FastTreeOva"] * 100).ToString(CultureInfo
                .InvariantCulture));
        Console.WriteLine(
            (bestMacroaccuracy["FastForestOva"] * 100).ToString(CultureInfo
                .InvariantCulture));
        Console.WriteLine(
            (bestMacroaccuracy["LbfgsMaximumEntropyMulti"] * 100).ToString(
                CultureInfo.InvariantCulture));
        Console.WriteLine(
            (bestMacroaccuracy["SdcaLogisticRegressionOva"] * 100).ToString(
                CultureInfo.InvariantCulture));
        Console.WriteLine(
            (bestMacroaccuracy["SdcaMaximumEntropyMulti"] * 100).ToString(
                CultureInfo.InvariantCulture));
        Console.WriteLine(
            (bestMacroaccuracy["LbfgsLogisticRegressionOva"] * 100).ToString(
                CultureInfo.InvariantCulture));
    }

    private void UpdateAndFilterAccuracies(Dictionary<string, float> accuracies,
        Dictionary<string, float> bestMacroaccuracy)
    {
        foreach (var accuracy in accuracies.Where(accuracy =>
                     !bestMacroaccuracy.ContainsKey(accuracy.Key) ||
                     bestMacroaccuracy[accuracy.Key] < accuracy.Value))
        {
            if (!accuracy.Key.Equals("FastTreeOva") &&
                !accuracy.Key.Equals("LbfgsMaximumEntropyMulti") &&
                !accuracy.Key.Equals("FastForestOva") &&
                !accuracy.Key.Equals("SdcaLogisticRegressionOva") &&
                !accuracy.Key.Equals("LbfgsLogisticRegressionOva") &&
                !accuracy.Key.Equals("SdcaMaximumEntropyMulti")) continue;
            bestMacroaccuracy[accuracy.Key] = accuracy.Value;
        }
    }
}