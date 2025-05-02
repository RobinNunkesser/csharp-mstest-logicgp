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
public class IrisTests : RealTests
{
    private readonly IDataView _data;

    private readonly LookupMap<string>[] _lookupData =
    [
        new("Iris-setosa"),
        new("Iris-versicolor"),
        new("Iris-virginica")
    ];

    public IrisTests()
    {
        ThreadSafeRandomNetCore.Seed = 42;
        ThreadSafeMLContext.Seed = 42;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
            "Data/Real/Iris", "Iris.csv");
        _data = mlContext.Data.LoadFromTextFile<IrisModelInput>(
            path,
            ',', true);
        LogFile = $"log_{GetType().Name}";
        //SaveCvSplit(_data, GetType().Name);
    }

    [TestCleanup]
    public void TearDown()
    {
        LogWriter?.Dispose();
    }

    [TestMethod]
    public void SimulateFlRwMacro()
    {
        var trainer = GetFlRwMacroTrainer(_lookupData.Length);
        SimulateFlRw(trainer, _data, _lookupData);
    }

    [TestMethod]
    public void SimulateMLNet()
    {
        SimulateMLNetOnAllTrainers(DataHelper.DataSet.Iris,
            "Data/Real/Iris", "Iris",
            "class", 20, true);
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


        Assert.IsTrue(metrics.MacroAccuracy > 0.559);
        Assert.IsTrue(metrics.MacroAccuracy < 0.56);
    }

    protected override EstimatorChain<ITransformer> GetPipeline(
        LogicGpTrainerBase<ITransformer> trainer, IDataView lookupIdvMap)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;

        var pipeline = mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"sepal length", @"sepal length"),
                new InputOutputColumnPair(@"sepal width", @"sepal width"),
                new InputOutputColumnPair(@"petal length", @"petal length"),
                new InputOutputColumnPair(@"petal width", @"petal width")
            })
            .Append(mlContext.Transforms.NormalizeBinning(new[]
            {
                new InputOutputColumnPair(@"sepal length", @"sepal length"),
                new InputOutputColumnPair(@"sepal width", @"sepal width"),
                new InputOutputColumnPair(@"petal length", @"petal length"),
                new InputOutputColumnPair(@"petal width", @"petal width")
            }, maximumBinCount: 4))
            .Append(mlContext.Transforms.Concatenate(@"Features",
                @"sepal length", @"sepal width", @"petal length",
                @"petal width"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(@"Label",
                @"class", keyData: lookupIdvMap))
            .Append(trainer);

        return pipeline;
    }
}