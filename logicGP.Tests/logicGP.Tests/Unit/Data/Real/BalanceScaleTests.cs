using Italbytz.AI;
using Italbytz.EA.Trainer;
using Italbytz.ML;
using Italbytz.ML.Data;
using logicGP.Tests.Data.Real;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace logicGP.Tests.Unit.Data.Real;

[TestClass]
public class BalanceScaleTests : RealTests
{
    private readonly IDataView _data;
    private readonly IDataset _dataset;

    public BalanceScaleTests()
    {
        _dataset = Italbytz.ML.Data.Data.BalanceScale;
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
            new LookupMap<string>("B"),
            new LookupMap<string>("R"),
            new LookupMap<string>("L")
        };
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var testResults = TestFlRw(trainer, _data, _data, lookupData, 10);
        var metrics = mlContext.MulticlassClassification
            .Evaluate(testResults);


        Assert.IsTrue(metrics.MacroAccuracy > 0.41);
        Assert.IsTrue(metrics.MacroAccuracy < 0.42);
    }

    protected override EstimatorChain<ITransformer> GetPipeline(
        IEstimator<ITransformer> trainer, IDataView lookupIdvMap)
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