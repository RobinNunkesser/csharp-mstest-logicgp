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
public class BreastCancerWisconsinDiagnosticTests : RealTests
{
    private readonly IDataView _data;

    private readonly LookupMap<string>[] _lookupData =
    [
        new("M"),
        new("B")
    ];

    public BreastCancerWisconsinDiagnosticTests()
    {
        ThreadSafeRandomNetCore.Seed = 42;
        ThreadSafeMLContext.Seed = 42;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
            "Data/Real/BreastCancerWisconsinDiagnostic",
            "Breast_Cancer_Wisconsin_Diagnostic_.csv");
        _data = mlContext.Data
            .LoadFromTextFile<BreastCancerWisconsinDiagnosticModelInput>(
                path,
                ',', true);
        LogFile = $"log_{GetType().Name}";
    }

    [TestCleanup]
    public void TearDown()
    {
        LogWriter?.Dispose();
    }

    [TestMethod]
    public void SimulateMLNet()
    {
        SimulateMLNetOnAllTrainers(
            DataHelper.DataSet.BreastCancerWisconsinDiagnostic,
            "Data/Real/BreastCancerWisconsinDiagnostic",
            "Breast_Cancer_Wisconsin_Diagnostic_",
            "Diagnosis", 20, false);
    }

    [TestMethod]
    public void SimulateFlRwMicro()
    {
        var trainer = GetFlRwMicroTrainer(_lookupData.Length);
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
                .GetRequiredService<LogicGpGpasBinaryTrainer>();

        trainer.Classes = _lookupData.Length;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var testResults = TestFlRw(trainer, _data, _data, _lookupData, 10);
        var metrics = mlContext.BinaryClassification
            .Evaluate(testResults, trainer.Label);


        Assert.IsTrue(metrics.Accuracy > 0.77);
        Assert.IsTrue(metrics.Accuracy < 0.78);
        Assert.IsTrue(metrics.F1Score > 0.79);
        Assert.IsTrue(metrics.F1Score < 0.80);
        Assert.IsTrue(metrics.AreaUnderRocCurve > 0.8);
        Assert.IsTrue(metrics.AreaUnderRocCurve < 0.81);
        Assert.IsTrue(metrics.AreaUnderPrecisionRecallCurve > 0.84);
        Assert.IsTrue(metrics.AreaUnderPrecisionRecallCurve < 0.85);
    }


    protected override EstimatorChain<ITransformer> GetPipeline(
        LogicGpTrainerBase<ITransformer> trainer, IDataView lookupIdvMap)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var pipeline = mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"radius1", @"radius1"),
                new InputOutputColumnPair(@"texture1", @"texture1"),
                new InputOutputColumnPair(@"perimeter1", @"perimeter1"),
                new InputOutputColumnPair(@"area1", @"area1"),
                new InputOutputColumnPair(@"smoothness1", @"smoothness1"),
                new InputOutputColumnPair(@"compactness1", @"compactness1"),
                new InputOutputColumnPair(@"concavity1", @"concavity1"),
                new InputOutputColumnPair(@"concave_points1",
                    @"concave_points1"),
                new InputOutputColumnPair(@"symmetry1", @"symmetry1"),
                new InputOutputColumnPair(@"fractal_dimension1",
                    @"fractal_dimension1"),
                new InputOutputColumnPair(@"radius2", @"radius2"),
                new InputOutputColumnPair(@"texture2", @"texture2"),
                new InputOutputColumnPair(@"perimeter2", @"perimeter2"),
                new InputOutputColumnPair(@"area2", @"area2"),
                new InputOutputColumnPair(@"smoothness2", @"smoothness2"),
                new InputOutputColumnPair(@"compactness2", @"compactness2"),
                new InputOutputColumnPair(@"concavity2", @"concavity2"),
                new InputOutputColumnPair(@"concave_points2",
                    @"concave_points2"),
                new InputOutputColumnPair(@"symmetry2", @"symmetry2"),
                new InputOutputColumnPair(@"fractal_dimension2",
                    @"fractal_dimension2"),
                new InputOutputColumnPair(@"radius3", @"radius3"),
                new InputOutputColumnPair(@"texture3", @"texture3"),
                new InputOutputColumnPair(@"perimeter3", @"perimeter3"),
                new InputOutputColumnPair(@"area3", @"area3"),
                new InputOutputColumnPair(@"smoothness3", @"smoothness3"),
                new InputOutputColumnPair(@"compactness3", @"compactness3"),
                new InputOutputColumnPair(@"concavity3", @"concavity3"),
                new InputOutputColumnPair(@"concave_points3",
                    @"concave_points3"),
                new InputOutputColumnPair(@"symmetry3", @"symmetry3"),
                new InputOutputColumnPair(@"fractal_dimension3",
                    @"fractal_dimension3")
            })
            .Append(mlContext.Transforms.NormalizeBinning(new[]
            {
                new InputOutputColumnPair(@"radius1", @"radius1"),
                new InputOutputColumnPair(@"texture1", @"texture1"),
                new InputOutputColumnPair(@"perimeter1", @"perimeter1"),
                new InputOutputColumnPair(@"area1", @"area1"),
                new InputOutputColumnPair(@"smoothness1", @"smoothness1"),
                new InputOutputColumnPair(@"compactness1", @"compactness1"),
                new InputOutputColumnPair(@"concavity1", @"concavity1"),
                new InputOutputColumnPair(@"concave_points1",
                    @"concave_points1"),
                new InputOutputColumnPair(@"symmetry1", @"symmetry1"),
                new InputOutputColumnPair(@"fractal_dimension1",
                    @"fractal_dimension1"),
                new InputOutputColumnPair(@"radius2", @"radius2"),
                new InputOutputColumnPair(@"texture2", @"texture2"),
                new InputOutputColumnPair(@"perimeter2", @"perimeter2"),
                new InputOutputColumnPair(@"area2", @"area2"),
                new InputOutputColumnPair(@"smoothness2", @"smoothness2"),
                new InputOutputColumnPair(@"compactness2", @"compactness2"),
                new InputOutputColumnPair(@"concavity2", @"concavity2"),
                new InputOutputColumnPair(@"concave_points2",
                    @"concave_points2"),
                new InputOutputColumnPair(@"symmetry2", @"symmetry2"),
                new InputOutputColumnPair(@"fractal_dimension2",
                    @"fractal_dimension2"),
                new InputOutputColumnPair(@"radius3", @"radius3"),
                new InputOutputColumnPair(@"texture3", @"texture3"),
                new InputOutputColumnPair(@"perimeter3", @"perimeter3"),
                new InputOutputColumnPair(@"area3", @"area3"),
                new InputOutputColumnPair(@"smoothness3", @"smoothness3"),
                new InputOutputColumnPair(@"compactness3", @"compactness3"),
                new InputOutputColumnPair(@"concavity3", @"concavity3"),
                new InputOutputColumnPair(@"concave_points3",
                    @"concave_points3"),
                new InputOutputColumnPair(@"symmetry3", @"symmetry3"),
                new InputOutputColumnPair(@"fractal_dimension3",
                    @"fractal_dimension3")
            }, maximumBinCount: 4))
            .Append(mlContext.Transforms.Concatenate(@"Features", @"radius1",
                @"texture1", @"perimeter1", @"area1", @"smoothness1",
                @"compactness1", @"concavity1", @"concave_points1",
                @"symmetry1", @"fractal_dimension1", @"radius2", @"texture2",
                @"perimeter2", @"area2", @"smoothness2", @"compactness2",
                @"concavity2", @"concave_points2", @"symmetry2",
                @"fractal_dimension2", @"radius3", @"texture3", @"perimeter3",
                @"area3", @"smoothness3", @"compactness3", @"concavity3",
                @"concave_points3", @"symmetry3", @"fractal_dimension3"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(@"Label",
                @"Diagnosis", keyData: lookupIdvMap))
            .Append(trainer);

        return pipeline;
    }
}