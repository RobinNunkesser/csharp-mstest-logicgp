using Italbytz.AI;
using Italbytz.EA.Trainer;
using Italbytz.ML;
using Italbytz.ML.Data;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace logicGP.Tests;

[TestClass]
public class LensesTests : RealTests
{
    private readonly IDataView _data;
    private readonly IDataset _dataset;

    public LensesTests()
    {
        _dataset = Italbytz.ML.Data.Data.Lenses;
        _data = _dataset.DataView;
    }

    [TestMethod]
    [TestCategory("FixedSeed")]
    public void TestFlRw()
    {
        ThreadSafeRandomNetCore.Seed = 42;

        var trainer = new LogicGpFlrwMacroMulticlassTrainer<TernaryClassificationOutput>(10);

        var lookupData = new[]
        {
            new LookupMap<uint>(1),
            new LookupMap<uint>(2),
            new LookupMap<uint>(3)
        };
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var testResults = TestFlRw(trainer, _data, _data, lookupData, 10);
        var metrics = mlContext.MulticlassClassification
            .Evaluate(testResults);

        Assert.IsTrue(metrics.MacroAccuracy > 0.8000);
        Assert.IsTrue(metrics.MacroAccuracy < 0.8001);
    }

    protected override EstimatorChain<ITransformer?> GetPipeline(
        IEstimator<ITransformer> trainer, IDataView lookupIdvMap)
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