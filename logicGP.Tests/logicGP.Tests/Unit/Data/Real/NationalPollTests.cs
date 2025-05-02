using Italbytz.Adapters.Algorithms.AI.Search.GP;
using Italbytz.Adapters.Algorithms.AI.Util;
using Italbytz.ML;
using logicGP.Tests.Data.Real;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace logicGP.Tests;

[TestClass]
public class NationalPollTests : RealTests
{
    private readonly IDataView _data;

    private readonly LookupMap<uint>[] _lookupData =
    [
        new(1),
        new(2),
        new(3)
    ];

    public NationalPollTests()
    {
        ThreadSafeRandomNetCore.Seed = 42;
        ThreadSafeMLContext.Seed = 42;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
            "Data/Real/NationalPoll",
            "national_poll_on_healthy_aging_npha.csv");
        _data = mlContext.Data.LoadFromTextFile<NationalPollModelInput>(
            path,
            ',', true);
        LogFile = $"log_{GetType().Name}";
        SaveCvSplit(_data, GetType().Name);
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
    public void TestFlRwMacro()
    {
        var trainer = GetFlRwMacroTrainer(_lookupData.Length);
        var testResults = TestFlRw(trainer, _data, _data, _lookupData, 10);
        var metrics = ThreadSafeMLContext.LocalMLContext
            .MulticlassClassification
            .Evaluate(testResults, trainer.Label);

        Assert.IsTrue(metrics.MacroAccuracy > 0.358);
        Assert.IsTrue(metrics.MacroAccuracy < 0.359);
    }


    protected override EstimatorChain<ITransformer?> GetPipeline(
        LogicGpTrainerBase<ITransformer> trainer, IDataView lookupIdvMap)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var pipeline = mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"Age", @"Age"),
                new InputOutputColumnPair(@"Physical_Health",
                    @"Physical_Health"),
                new InputOutputColumnPair(@"Mental_Health", @"Mental_Health"),
                new InputOutputColumnPair(@"Dental_Health", @"Dental_Health"),
                new InputOutputColumnPair(@"Employment", @"Employment"),
                new InputOutputColumnPair(@"Stress_Keeps_Patient_from_Sleeping",
                    @"Stress_Keeps_Patient_from_Sleeping"),
                new InputOutputColumnPair(
                    @"Medication_Keeps_Patient_from_Sleeping",
                    @"Medication_Keeps_Patient_from_Sleeping"),
                new InputOutputColumnPair(@"Pain_Keeps_Patient_from_Sleeping",
                    @"Pain_Keeps_Patient_from_Sleeping"),
                new InputOutputColumnPair(
                    @"Bathroom_Needs_Keeps_Patient_from_Sleeping",
                    @"Bathroom_Needs_Keeps_Patient_from_Sleeping"),
                new InputOutputColumnPair(@"Uknown_Keeps_Patient_from_Sleeping",
                    @"Uknown_Keeps_Patient_from_Sleeping"),
                new InputOutputColumnPair(@"Trouble_Sleeping",
                    @"Trouble_Sleeping"),
                new InputOutputColumnPair(@"Prescription_Sleep_Medication",
                    @"Prescription_Sleep_Medication"),
                new InputOutputColumnPair(@"Race", @"Race"),
                new InputOutputColumnPair(@"Gender", @"Gender")
            })
            .Append(mlContext.Transforms.Concatenate(@"Features", @"Age",
                @"Physical_Health", @"Mental_Health", @"Dental_Health",
                @"Employment", @"Stress_Keeps_Patient_from_Sleeping",
                @"Medication_Keeps_Patient_from_Sleeping",
                @"Pain_Keeps_Patient_from_Sleeping",
                @"Bathroom_Needs_Keeps_Patient_from_Sleeping",
                @"Uknown_Keeps_Patient_from_Sleeping", @"Trouble_Sleeping",
                @"Prescription_Sleep_Medication", @"Race", @"Gender"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(
                @"Label", @"Number_of_Doctors_Visited",
                keyData: lookupIdvMap))
            .Append(trainer);

        return pipeline;
    }
}