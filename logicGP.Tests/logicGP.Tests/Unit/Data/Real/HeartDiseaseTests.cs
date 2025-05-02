using Italbytz.Adapters.Algorithms.AI.Search.GP;
using Italbytz.Adapters.Algorithms.AI.Util;
using Italbytz.ML;
using logicGP.Tests.Data.Real;
using logicGP.Tests.Util;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace logicGP.Tests.Unit.Data.Real;

[TestClass]
public class HeartDiseaseTests : RealTests
{
    private readonly IDataView _data;

    private readonly LookupMap<uint>[] _lookupData =
    [
        new(0),
        new(1),
        new(2),
        new(3),
        new(4)
    ];

    public HeartDiseaseTests()
    {
        ThreadSafeRandomNetCore.Seed = 42;
        ThreadSafeMLContext.Seed = 42;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
            "Data/Real/HeartDisease", "Heart_Disease.csv");
        _data = mlContext.Data.LoadFromTextFile<HeartDiseaseModelInput>(
            path,
            ',', true);
        LogFile = $"log_{GetType().Name}";
        //SaveTrainTestSplit(_data, GetType().Name);
    }

    [TestCleanup]
    public void TearDown()
    {
        LogWriter?.Dispose();
    }

    [TestMethod]
    public void SimulateMLNet()
    {
        SimulateMLNetOnAllTrainers(DataHelper.DataSet.HeartDisease,
            "Data/Real/HeartDisease", "Heart_Disease",
            "num", 20, true);
    }

    [TestMethod]
    public void SimulateFlRwMacro()
    {
        var trainer = GetFlRwMacroTrainer(_lookupData.Length);
        SimulateFlRw(trainer, _data, _lookupData);
    }

    [TestMethod]
    [TestCategory("FixedSeed")]
    public void TestFlRw()
    {
        ThreadSafeRandomNetCore.Seed = 42;

        var services = new ServiceCollection().AddServices();
        var serviceProvider = services.BuildServiceProvider();
        var trainer =
            serviceProvider
                .GetRequiredService<LogicGpFlrwMacroMulticlassTrainer>();
        trainer.Classes = _lookupData.Length;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var testResults = TestFlRw(trainer, _data, _data, _lookupData, 10);
        var metrics = mlContext.MulticlassClassification
            .Evaluate(testResults, trainer.Label);


        Assert.IsTrue(metrics.MacroAccuracy > 0.24);
        Assert.IsTrue(metrics.MacroAccuracy < 0.25);
    }


    protected override EstimatorChain<ITransformer> GetPipeline(
        LogicGpTrainerBase<ITransformer> trainer, IDataView lookupIdvMap)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var pipeline = mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"age", @"age"),
                new InputOutputColumnPair(@"sex", @"sex"),
                new InputOutputColumnPair(@"cp", @"cp"),
                new InputOutputColumnPair(@"trestbps", @"trestbps"),
                new InputOutputColumnPair(@"chol", @"chol"),
                new InputOutputColumnPair(@"fbs", @"fbs"),
                new InputOutputColumnPair(@"restecg", @"restecg"),
                new InputOutputColumnPair(@"thalach", @"thalach"),
                new InputOutputColumnPair(@"exang", @"exang"),
                new InputOutputColumnPair(@"oldpeak", @"oldpeak"),
                new InputOutputColumnPair(@"slope", @"slope"),
                new InputOutputColumnPair(@"ca", @"ca"),
                new InputOutputColumnPair(@"thal", @"thal")
            })
            .Append(mlContext.Transforms.NormalizeBinning(new[]
            {
                new InputOutputColumnPair(@"age", @"age"),
                new InputOutputColumnPair(@"trestbps", @"trestbps"),
                new InputOutputColumnPair(@"chol", @"chol"),
                new InputOutputColumnPair(@"thalach", @"thalach"),
                new InputOutputColumnPair(@"oldpeak", @"oldpeak"),
                new InputOutputColumnPair(@"ca", @"ca")
            }, maximumBinCount: 4))
            .Append(mlContext.Transforms.Concatenate(@"Features", @"age",
                @"sex", @"cp", @"trestbps", @"chol", @"fbs", @"restecg",
                @"thalach", @"exang", @"oldpeak", @"slope", @"ca", @"thal"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(@"Label",
                @"num", keyData: lookupIdvMap))
            .Append(trainer);

        return pipeline;
    }
}