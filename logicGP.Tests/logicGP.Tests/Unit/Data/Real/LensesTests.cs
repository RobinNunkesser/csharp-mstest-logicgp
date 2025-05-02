using Italbytz.Adapters.Algorithms.AI.Search.GP;
using Italbytz.Adapters.Algorithms.AI.Util;
using Italbytz.ML;
using logicGP.Tests.Data.Real;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace logicGP.Tests;

[TestClass]
public class LensesTests : RealTests
{
    private readonly IDataView _data;

    public LensesTests()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
            "Data/Real/Lenses", "lenses.csv");
        _data = mlContext.Data.LoadFromTextFile<LensesModelInput>(
            path,
            ',', true);
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

        var lookupData = new[]
        {
            new LookupMap<uint>(1),
            new LookupMap<uint>(2),
            new LookupMap<uint>(3)
        };
        trainer.Classes = lookupData.Length;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var testResults = TestFlRw(trainer, _data, _data, lookupData, 10);
        var metrics = mlContext.MulticlassClassification
            .Evaluate(testResults, trainer.Label);

        Assert.IsTrue(metrics.MacroAccuracy > 0.8000);
        Assert.IsTrue(metrics.MacroAccuracy < 0.8001);
    }

    protected override EstimatorChain<ITransformer?> GetPipeline(
        LogicGpTrainerBase<ITransformer> trainer, IDataView lookupIdvMap)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var pipeline = mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"age", @"age"),
                new InputOutputColumnPair(@"spectacle_prescription",
                    @"spectacle_prescription"),
                new InputOutputColumnPair(@"astigmatic", @"astigmatic")
            })
            .Append(mlContext.Transforms.Concatenate(@"Features", @"age",
                @"spectacle_prescription", @"astigmatic"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(@"Label",
                @"class", keyData: lookupIdvMap))
            .Append(trainer);

        return pipeline;
    }
}