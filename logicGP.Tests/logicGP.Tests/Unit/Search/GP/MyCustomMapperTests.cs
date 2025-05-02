using Italbytz.Adapters.Algorithms.AI.Search.GP;
using Italbytz.ML;
using logicGP.Tests.Data.Real;
using logicGP.Tests.Data.Simulated;
using Microsoft.ML;
using Microsoft.ML.Data;
using InputOutputColumnPair = Microsoft.ML.InputOutputColumnPair;

namespace logicGP.Tests.Unit.Search.GP;

[TestClass]
public class MyCustomMapperTests
{
    [TestMethod]
    public void VectorDataTypeTests()
    {
        var score = new VBuffer<float>(3, 3, [1, 2, 3],
            [0, 1, 2]);
        var test = new VectorDataViewType(NumberDataViewType.Single, 3);
        Assert.AreEqual(3, test.Size);
    }

    [TestMethod]
    public void CustomMulticlassMappingTrainerTest()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var trainDataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
            "Data/Real", "balancescale.csv");
        var trainData =
            mlContext.Data.LoadFromTextFile<BalanceScaleModelInput>(
                trainDataPath,
                ',', true);

        var lookupData = new[]
        {
            new LookupMap<string>("B"),
            new LookupMap<string>("R"),
            new LookupMap<string>("L")
        };
        // Convert to IDataView
        var lookupIdvMap = mlContext.Data.LoadFromEnumerable(lookupData);

        var pipeline = mlContext.Transforms.ReplaceMissingValues(new[]
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
                @"left-weight")).Append(new MyCustomMulticlassEstimator());
        var model = pipeline.Fit(trainData);
        var transformedData = model.Transform(trainData);
        var metrics = mlContext.MulticlassClassification
            .Evaluate(transformedData);
        Assert.IsNotNull(metrics);
    }

    [TestMethod]
    public void CustomBinaryMappingTrainerTest()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var trainDataPath = Path.Combine(
            AppDomain.CurrentDomain.BaseDirectory,
            "Data/Simulated/Simulation1", "SNPglm_1.csv");
        var trainData = mlContext.Data.LoadFromTextFile<SNPModelInput>(
            trainDataPath,
            ',', true);

        var lookupData = new[]
        {
            new LookupMap<uint>(0),
            new LookupMap<uint>(1)
        };


        // Convert to IDataView
        var lookupIdvMap = mlContext.Data.LoadFromEnumerable(lookupData);


        // Data process configuration with pipeline data transformations
        var pipeline = mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"SNP1", @"SNP1"),
                new InputOutputColumnPair(@"SNP2", @"SNP2"),
                new InputOutputColumnPair(@"SNP3", @"SNP3"),
                new InputOutputColumnPair(@"SNP4", @"SNP4"),
                new InputOutputColumnPair(@"SNP5", @"SNP5"),
                new InputOutputColumnPair(@"SNP6", @"SNP6"),
                new InputOutputColumnPair(@"SNP7", @"SNP7"),
                new InputOutputColumnPair(@"SNP8", @"SNP8"),
                new InputOutputColumnPair(@"SNP9", @"SNP9"),
                new InputOutputColumnPair(@"SNP10", @"SNP10"),
                new InputOutputColumnPair(@"SNP11", @"SNP11"),
                new InputOutputColumnPair(@"SNP12", @"SNP12"),
                new InputOutputColumnPair(@"SNP13", @"SNP13"),
                new InputOutputColumnPair(@"SNP14", @"SNP14"),
                new InputOutputColumnPair(@"SNP15", @"SNP15"),
                new InputOutputColumnPair(@"SNP16", @"SNP16"),
                new InputOutputColumnPair(@"SNP17", @"SNP17"),
                new InputOutputColumnPair(@"SNP18", @"SNP18"),
                new InputOutputColumnPair(@"SNP19", @"SNP19"),
                new InputOutputColumnPair(@"SNP20", @"SNP20"),
                new InputOutputColumnPair(@"SNP21", @"SNP21"),
                new InputOutputColumnPair(@"SNP22", @"SNP22"),
                new InputOutputColumnPair(@"SNP23", @"SNP23"),
                new InputOutputColumnPair(@"SNP24", @"SNP24"),
                new InputOutputColumnPair(@"SNP25", @"SNP25"),
                new InputOutputColumnPair(@"SNP26", @"SNP26"),
                new InputOutputColumnPair(@"SNP27", @"SNP27"),
                new InputOutputColumnPair(@"SNP28", @"SNP28"),
                new InputOutputColumnPair(@"SNP29", @"SNP29"),
                new InputOutputColumnPair(@"SNP30", @"SNP30"),
                new InputOutputColumnPair(@"SNP31", @"SNP31"),
                new InputOutputColumnPair(@"SNP32", @"SNP32"),
                new InputOutputColumnPair(@"SNP33", @"SNP33"),
                new InputOutputColumnPair(@"SNP34", @"SNP34"),
                new InputOutputColumnPair(@"SNP35", @"SNP35"),
                new InputOutputColumnPair(@"SNP36", @"SNP36"),
                new InputOutputColumnPair(@"SNP37", @"SNP37"),
                new InputOutputColumnPair(@"SNP38", @"SNP38"),
                new InputOutputColumnPair(@"SNP39", @"SNP39"),
                new InputOutputColumnPair(@"SNP40", @"SNP40"),
                new InputOutputColumnPair(@"SNP41", @"SNP41"),
                new InputOutputColumnPair(@"SNP42", @"SNP42"),
                new InputOutputColumnPair(@"SNP43", @"SNP43"),
                new InputOutputColumnPair(@"SNP44", @"SNP44"),
                new InputOutputColumnPair(@"SNP45", @"SNP45"),
                new InputOutputColumnPair(@"SNP46", @"SNP46"),
                new InputOutputColumnPair(@"SNP47", @"SNP47"),
                new InputOutputColumnPair(@"SNP48", @"SNP48"),
                new InputOutputColumnPair(@"SNP49", @"SNP49"),
                new InputOutputColumnPair(@"SNP50", @"SNP50")
            })
            .Append(mlContext.Transforms.Concatenate(@"Features", @"SNP1",
                @"SNP2", @"SNP3", @"SNP4", @"SNP5", @"SNP6", @"SNP7", @"SNP8",
                @"SNP9", @"SNP10", @"SNP11", @"SNP12", @"SNP13", @"SNP14",
                @"SNP15", @"SNP16", @"SNP17", @"SNP18", @"SNP19", @"SNP20",
                @"SNP21", @"SNP22", @"SNP23", @"SNP24", @"SNP25", @"SNP26",
                @"SNP27", @"SNP28", @"SNP29", @"SNP30", @"SNP31", @"SNP32",
                @"SNP33", @"SNP34", @"SNP35", @"SNP36", @"SNP37", @"SNP38",
                @"SNP39", @"SNP40", @"SNP41", @"SNP42", @"SNP43", @"SNP44",
                @"SNP45", @"SNP46", @"SNP47", @"SNP48", @"SNP49", @"SNP50"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(@"Label",
                @"y",
                keyData: lookupIdvMap))
            .Append(new MyCustomBinaryEstimator());
        /*.Append(
            mlContext.Transforms.Conversion.MapKeyToValue(@"PredictedLabel",
                @"PredictedLabel"));*/


        var model = pipeline.Fit(trainData);
        var transformedData = model.Transform(trainData);
        var metrics = mlContext.BinaryClassification
            .Evaluate(transformedData);
        Assert.IsNotNull(metrics);
    }
}

// Type for the IDataView that will be serving as the map