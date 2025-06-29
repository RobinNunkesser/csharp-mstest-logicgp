using System.Globalization;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Control;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Fitness;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Selection;
using Italbytz.AI.Search.GP.Individuals;
using Italbytz.AI.Search.GP.Selection;
using Italbytz.ML;
using Microsoft.ML;
using Microsoft.ML.Data;
using DefaultColumnNames = Italbytz.ML.DefaultColumnNames;

namespace Italbytz.Adapters.Algorithms.AI.Search.GP;

public abstract class
    LogicGpTrainerBase<TTransformer>(
        LogicGpAlgorithm algorithm,
        DataManager data) : IEstimator<ITransformer>


{
    public IIndividual? ChosenIndividual { get; set; }
    public required string Label { get; set; }
    public required int MaxGenerations { get; set; } = 10000;
    public required int Classes { get; set; } = 2;

    public required Dictionary<int, string> LabelKeyToValueDictionary
    {
        get;
        set;
    }

    public ITransformer Fit(IDataView input)
    {
        /*var de = input.GetDataExcerpt();
        foreach (var feature in de.Features)
            Console.WriteLine(string.Join(",",feature.Select(f => f.ToString(CultureInfo.InvariantCulture))));
        Console.Out.Flush();*/
                
        var featureNames = input.GetFeaturesSlotNames();
        // Split data into k folds
        const int k = 5; // Number of folds
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var cvResults = mlContext.Data.CrossValidationSplit(input);
        var candidates = new IIndividualList[k];
        var foldIndex = 0;

        ParameterizeAlgorithm(algorithm);
        foreach (var fold in cvResults)
        {
            var trainSet = fold.TrainSet;
            var testSet = fold.TestSet;
            var trainFeatures = trainSet
                .GetColumn<float[]>(DefaultColumnNames.Features)
                .ToList();
            // Training
            var individuals =
                algorithm.Train(trainSet, foldIndex == 0, Label,
                    MaxGenerations);
            // Validating
            var validationMetrics =
                algorithm.Validate(testSet, individuals, Label);
            // Selecting
            var selection = new BestModelForEachSizeSelection();
            candidates[foldIndex++] = selection.Process(individuals);
        }

        // Final Selection
        var bestSelection = new FinalModelSelection();
        var allCandidates = candidates.SelectMany(i => i).ToList();
        var candidatePopulation = new Population();
        foreach (var candidate in allCandidates)
            candidatePopulation.Add(candidate);
        ChosenIndividual = bestSelection.Process(candidatePopulation)[0];
        Console.WriteLine($"Chosen individual: \n{ChosenIndividual}");


        var mapping = new LogicGpMapping(ChosenIndividual);
        return Classes switch
        {
            2 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<ClassificationInput,
                            BinaryClassificationOutput>(), null)
                .Fit(input),
            3 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<ClassificationInput,
                            TernaryClassificationOutput>(),
                    null)
                .Fit(input),
            4 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<ClassificationInput,
                            QuaternaryClassificationOutput>(),
                    null)
                .Fit(input),
            5 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<ClassificationInput,
                            QuinaryClassificationOutput>(),
                    null)
                .Fit(input),
            6 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<ClassificationInput,
                            SenaryClassificationOutput>(),
                    null)
                .Fit(input),
            7 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<ClassificationInput,
                            SeptenaryClassificationOutput>(),
                    null)
                .Fit(input),
            8 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<ClassificationInput,
                            OctonaryClassificationOutput>(),
                    null)
                .Fit(input),
            9 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<ClassificationInput,
                            NonaryClassificationOutput>(),
                    null)
                .Fit(input),
            10 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<ClassificationInput,
                            DenaryClassificationOutput>(),
                    null)
                .Fit(input),
            11 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<ClassificationInput,
                            UndenaryClassificationOutput>(),
                    null)
                .Fit(input),
            12 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<ClassificationInput,
                            DuodenaryClassificationOutput>(),
                    null)
                .Fit(input),
            13 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<ClassificationInput,
                            TridenaryClassificationOutput>(),
                    null)
                .Fit(input),
            14 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<ClassificationInput,
                            TetradenaryClassificationOutput>(),
                    null)
                .Fit(input),
            15 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<ClassificationInput,
                            PentadenaryClassificationOutput>(),
                    null)
                .Fit(input),
            _ => throw new ArgumentOutOfRangeException(
                $"The number of classes {Classes} is not supported.")
        };
    }

    /// <summary>
    ///     This method cannot be implemented with reasonable effort because
    ///     ML.NET only exposes the necessary API to "best friends".
    /// </summary>
    /// <param name="inputSchema"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var mapping = new LogicGpMapping(ChosenIndividual);
        if (Classes == 2)
            return mlContext.Transforms.CustomMapping(
                mapping
                    .GetMapping<ClassificationInput,
                        BinaryClassificationOutput>(),
                null).GetOutputSchema(inputSchema);

        return mlContext.Transforms.CustomMapping(
            mapping
                .GetMapping<ClassificationInput,
                    ClassificationInput>(),
            null).GetOutputSchema(inputSchema);
    }


    protected abstract void ParameterizeAlgorithm(
        LogicGpAlgorithm logicGpAlgorithm);
}