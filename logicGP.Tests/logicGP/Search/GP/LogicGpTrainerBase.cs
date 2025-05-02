using Italbytz.Adapters.Algorithms.AI.Search.GP.Control;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Fitness;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Individuals;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Selection;
using Italbytz.ML;
using Italbytz.Ports.Algorithms.AI.Search.GP.Individuals;
using Microsoft.ML;
using Microsoft.ML.Data;

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
                        .GetMapping<BinaryClassificationInputSchema,
                            BinaryClassificationOutputSchema>(), null)
                .Fit(input),
            3 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<MulticlassClassificationInputSchema,
                            TernaryClassificationClassificationOutputSchema>(),
                    null)
                .Fit(input),
            4 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<MulticlassClassificationInputSchema,
                            QuaternaryClassificationClassificationOutputSchema>(),
                    null)
                .Fit(input),
            5 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<MulticlassClassificationInputSchema,
                            QuinaryClassificationClassificationOutputSchema>(),
                    null)
                .Fit(input),
            6 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<MulticlassClassificationInputSchema,
                            SenaryClassificationClassificationOutputSchema>(),
                    null)
                .Fit(input),
            7 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<MulticlassClassificationInputSchema,
                            SeptenaryClassificationClassificationOutputSchema>(),
                    null)
                .Fit(input),
            8 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<MulticlassClassificationInputSchema,
                            OctonaryClassificationClassificationOutputSchema>(),
                    null)
                .Fit(input),
            9 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<MulticlassClassificationInputSchema,
                            NonaryClassificationClassificationOutputSchema>(),
                    null)
                .Fit(input),
            10 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<MulticlassClassificationInputSchema,
                            DenaryClassificationClassificationOutputSchema>(),
                    null)
                .Fit(input),
            11 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<MulticlassClassificationInputSchema,
                            UndenaryClassificationClassificationOutputSchema>(),
                    null)
                .Fit(input),
            12 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<MulticlassClassificationInputSchema,
                            DuodenaryClassificationClassificationOutputSchema>(),
                    null)
                .Fit(input),
            13 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<MulticlassClassificationInputSchema,
                            TridenaryClassificationClassificationOutputSchema>(),
                    null)
                .Fit(input),
            14 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<MulticlassClassificationInputSchema,
                            TetradenaryClassificationClassificationOutputSchema>(),
                    null)
                .Fit(input),
            15 => mlContext.Transforms.CustomMapping(
                    mapping
                        .GetMapping<MulticlassClassificationInputSchema,
                            PentadenaryClassificationClassificationOutputSchema>(),
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
                    .GetMapping<BinaryClassificationInputSchema,
                        BinaryClassificationOutputSchema>(),
                null).GetOutputSchema(inputSchema);

        return mlContext.Transforms.CustomMapping(
            mapping
                .GetMapping<MulticlassClassificationInputSchema,
                    MulticlassClassificationInputSchema>(),
            null).GetOutputSchema(inputSchema);
    }


    protected abstract void ParameterizeAlgorithm(
        LogicGpAlgorithm logicGpAlgorithm);
}