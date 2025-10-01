using System.Diagnostics;
using Italbytz.AI;
using Italbytz.EA.Trainer;
using Italbytz.ML;
using Italbytz.ML.Data;
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

    private readonly IDataset _dataset;

    public NationalPollTests()
    {
        ThreadSafeRandomNetCore.Seed = 42;
        ThreadSafeMLContext.Seed = 42;
        _dataset = Italbytz.ML.Data.Data.NPHA;
        _data = _dataset.DataView;
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
        var trainer =
            new LogicGpFlcwMacroMulticlassTrainer<TernaryClassificationOutput>(
                10000);
        SimulateFlRw(trainer, _data, _lookupData);
    }

    [TestMethod]
    public void TestFlRwMacroRuntime()
    {
        var folder = AppDomain.CurrentDomain.BaseDirectory;
        var file = Path.Combine(folder, $"{GetType().Name}_runtime.csv");
        var resultWriter = new StreamWriter(file);
        resultWriter.WriteLine("Generations,TimeMs");
        var trainer = new LogicGpFlcwMacroMulticlassTrainer<TernaryClassificationOutput>(
            10000);
        var stopwatch = new Stopwatch();
        var generations = new int[] { 10000 };//{ 10, 100, 1000, 10000};
        var magicNumbers = new int[] { 42 };//{ 3, 7, 13, 21, 42, 64, 77, 88, 99, 123 };
        foreach (var generation in generations)
        {
            foreach (var magicNumber in magicNumbers)
            {
                ThreadSafeRandomNetCore.Seed = magicNumber;
                ThreadSafeMLContext.Seed = magicNumber;
                stopwatch.Start();
                TestFlRw(trainer, _data, _data, _lookupData, generation);
                stopwatch.Stop();
                resultWriter.WriteLine(
                    $"{generation},{stopwatch.ElapsedMilliseconds}");
                resultWriter.Flush();
                stopwatch.Reset();
            }
        }
        resultWriter.Dispose();
    }

    [TestMethod]
    [TestCategory("FixedSeed")]
    public void TestFlRwMacro()
    {
        var trainer = new LogicGpFlcwMacroMulticlassTrainer<TernaryClassificationOutput>(
            10);
        var testResults = TestFlRw(trainer, _data, _data, _lookupData, 10);
        var metrics = ThreadSafeMLContext.LocalMLContext
            .MulticlassClassification
            .Evaluate(testResults);

        Assert.IsTrue(metrics.MacroAccuracy > 0.33);
        Assert.IsTrue(metrics.MacroAccuracy < 0.34);
    }


    protected override EstimatorChain<ITransformer?> GetPipeline(
        IEstimator<ITransformer> trainer, IDataView lookupIdvMap)
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