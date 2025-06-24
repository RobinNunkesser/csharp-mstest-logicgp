using Italbytz.Adapters.Algorithms.AI.Search.GP;
using Italbytz.AI.Util;
using Italbytz.ML;
using Italbytz.ML.Data;
using Italbytz.ML.ModelBuilder.Configuration;
using logicGP.Tests.Data.Real;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace logicGP.Tests.Unit.Data.Real;

[TestClass]
public class IrisTests : RealTests
{
    private readonly IDataView _data;
    private readonly IDataset _dataset;
    
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
        _dataset = Italbytz.ML.Data.Data.Iris;
        _data = _dataset.DataView;
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

    protected override IEstimator<ITransformer> GetPipeline(
        LogicGpTrainerBase<ITransformer> trainer, IDataView lookupIdvMap)
    {
        return _dataset.BuildPipeline(
            ThreadSafeMLContext.LocalMLContext, ScenarioType.Classification, trainer,true);
    }
}