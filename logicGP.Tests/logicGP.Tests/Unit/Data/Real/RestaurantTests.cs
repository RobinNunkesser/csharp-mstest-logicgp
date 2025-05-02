using Italbytz.Adapters.Algorithms.AI.Search.GP;
using Italbytz.Adapters.Algorithms.AI.Util;
using Italbytz.ML;
using logicGP.Tests.Data.Real;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace logicGP.Tests;

[TestClass]
public class RestaurantTests : RealTests
{
    private readonly IDataView _data;

    public RestaurantTests()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
            "Data/Real/Restaurant", "restaurant.csv");
        _data = mlContext.Data.LoadFromTextFile<RestaurantModelInput>(
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
                .GetRequiredService<LogicGpGpasBinaryTrainer>();

        var lookupData = new[]
        {
            new LookupMap<uint>(0),
            new LookupMap<uint>(1)
        };
        trainer.Classes = lookupData.Length;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var testResults = TestFlRw(trainer, _data, _data, lookupData, 10);
        var metrics = mlContext.BinaryClassification
            .Evaluate(testResults, trainer.Label);

        Assert.IsTrue(metrics.Accuracy > 0.749);
        Assert.IsTrue(metrics.Accuracy < 0.751);
        Assert.IsTrue(metrics.F1Score > 0.76);
        Assert.IsTrue(metrics.F1Score < 0.77);
        Assert.IsTrue(metrics.AreaUnderRocCurve > 0.74);
        Assert.IsTrue(metrics.AreaUnderRocCurve < 0.76);
        Assert.IsTrue(metrics.AreaUnderPrecisionRecallCurve > 0.74);
        Assert.IsTrue(metrics.AreaUnderPrecisionRecallCurve < 0.75);
    }

    protected override EstimatorChain<ITransformer?> GetPipeline(
        LogicGpTrainerBase<ITransformer> trainer, IDataView lookupIdvMap)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var pipeline = mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"alternate", @"alternate"),
                new InputOutputColumnPair(@"bar", @"bar"),
                new InputOutputColumnPair(@"fri/sat", @"fri/sat"),
                new InputOutputColumnPair(@"hungry", @"hungry"),
                new InputOutputColumnPair(@"patrons", @"patrons"),
                new InputOutputColumnPair(@"price", @"price"),
                new InputOutputColumnPair(@"raining", @"raining"),
                new InputOutputColumnPair(@"reservation", @"reservation"),
                new InputOutputColumnPair(@"type", @"type"),
                new InputOutputColumnPair(@"wait_estimate", @"wait_estimate")
            })
            .Append(mlContext.Transforms.Concatenate(@"Features", @"alternate",
                @"bar", @"fri/sat", @"hungry", @"patrons", @"price", @"raining",
                @"reservation", @"type", @"wait_estimate"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(@"Label",
                @"will_wait", keyData: lookupIdvMap))
            .Append(trainer);

        return pipeline;
    }
}