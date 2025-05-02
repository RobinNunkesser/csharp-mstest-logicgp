using Italbytz.Adapters.Algorithms.AI.Search.GP;
using Italbytz.Adapters.Algorithms.AI.Util;
using Italbytz.ML;
using logicGP.Tests.Data.Real;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace logicGP.Tests.Unit.Data.Real;

[TestClass]
public class BalanceScaleTests : RealTests
{
    private readonly IDataView _data;

    public BalanceScaleTests()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
            "Data/Real/BalanceScale", "balancescale.csv");
        _data = mlContext.Data.LoadFromTextFile<BalanceScaleModelInput>(
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
            new LookupMap<string>("B"),
            new LookupMap<string>("R"),
            new LookupMap<string>("L")
        };
        trainer.Classes = lookupData.Length;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var testResults = TestFlRw(trainer, _data, _data, lookupData, 10);
        var metrics = mlContext.MulticlassClassification
            .Evaluate(testResults, trainer.Label);


        Assert.IsTrue(metrics.MacroAccuracy > 0.45);
        Assert.IsTrue(metrics.MacroAccuracy < 0.46);
    }

    protected override EstimatorChain<ITransformer> GetPipeline(
        LogicGpTrainerBase<ITransformer> trainer, IDataView lookupIdvMap)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        return mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"right-distance",
                    @"right-distance"),
                new InputOutputColumnPair(@"right-weight", @"right-weight"),
                new InputOutputColumnPair(@"left-distance",
                    @"left-distance"),
                new InputOutputColumnPair(@"left-weight", @"left-weight")
            })
            .Append(mlContext.Transforms.Conversion.MapValueToKey(@"Label",
                @"class", keyData: lookupIdvMap))
            .Append(mlContext.Transforms.Concatenate(@"Features",
                @"right-distance", @"right-weight", @"left-distance",
                @"left-weight")).Append(trainer);
    }
}