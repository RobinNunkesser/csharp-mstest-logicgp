using logicGP.Tests.Data.Real;
using Microsoft.ML;

namespace logicGP.Tests.Unit.Data.Real;

[TestClass]
public class HeartDiseaseEvaluations
{
    [TestMethod]
    public void EvaluateMLRuns()
    { 
        int[] seeds = 
        [
            42, 7, 13, 23, 3, 666, 777, 9, 17, 21
        ];
        string[] availableTrainers =
        [
            "LBFGS", "FASTFOREST", "SDCA", "FASTTREE"
        ];
        var baseDataDir = @"FOO";
        var baseModelDir =
            @"BAR";
        
        var labelColumn = "num";
        var mlContext = new MLContext();
        foreach (var trainer in availableTrainers)
        foreach (var seed in seeds)
        {
            var testData = Path.Combine(baseDataDir,$"Heart_Disease_seed_{seed}_test.csv");
            var modelPath = Path.Combine(baseModelDir,$"Heart_Disease_seed_{seed}_{trainer}.mlnet");

            try
            {
                var mlModel = mlContext.Model.Load(modelPath, out _);

                var testDataView = mlContext.Data
                    .LoadFromTextFile<HeartDiseaseModelInputOriginal>(
                        testData,
                        ',', true);
                var testResult = mlModel.Transform(testDataView);
                var metrics = mlContext.MulticlassClassification
                    .Evaluate(testResult, labelColumn);
                Console.WriteLine($"{trainer}({seed}): {metrics.MacroAccuracy}");
            }
            catch (Exception e)
            {
                Console.WriteLine(
                    $"Error loading model for data set {testData} and trainer {trainer}.");
            }
        }
    }
}