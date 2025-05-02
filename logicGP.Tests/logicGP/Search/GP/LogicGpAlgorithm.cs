using System.Collections;
using Italbytz.Adapters.Algorithms.AI.Search.Framework;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Control;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Crossover;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Fitness;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Initialization;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Mutation;
using Italbytz.Adapters.Algorithms.AI.Search.GP.PopulationManager;
using Italbytz.Adapters.Algorithms.AI.Search.GP.SearchSpace;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Selection;
using Italbytz.Adapters.Algorithms.AI.Search.GP.StoppingCriterion;
using Italbytz.ML;
using Italbytz.Ports.Algorithms.AI.Search;
using Italbytz.Ports.Algorithms.AI.Search.GP;
using Italbytz.Ports.Algorithms.AI.Search.GP.Fitness;
using Italbytz.Ports.Algorithms.AI.Search.GP.Individuals;
using Italbytz.Ports.Algorithms.AI.Search.GP.Mutation;
using Microsoft.ML;

namespace Italbytz.Adapters.Algorithms.AI.Search.GP;

/// <summary>
///     A class representing the LogicGP algorithm for genetic programming.
///     It provides methods for training,
///     validating, and testing the algorithm.
/// </summary>
/// <remarks>
///     The LogicGpAlgorithm class is used to represent the LogicGP algorithm for
///     genetic programming.
///     It contains methods for training, validating, and testing the algorithm.
///     The class also provides properties for setting the accuracy, weighting,
///     initialization, population manager, search space, stopping criterion,
///     selection, and fitness function.
///     The class also provides methods for preparing the algorithm for training
///     and
///     adapting literals.
/// </remarks>
public class LogicGpAlgorithm(
    IGeneticProgram gp,
    RandomInitialization randomInitialization,
    CompleteInitialization completeInitialization,
    DefaultPopulationManager populationManager,
    LogicGpSearchSpace searchSpace,
    GenerationStoppingCriterion generationStoppingCriterion,
    UniformSelection selection,
    ParetoFrontSelection paretoFrontSelection,
    IFitnessFunction fitnessFunction,
    DataManager data)
{
    public enum Accuracies
    {
        Macro,
        Micro
    }

    public enum PredictionStrategy
    {
        Max,
        SoftmaxProbability
    }

    public enum Weighting
    {
        Fixed,
        Computed,
        Mutated
    }


    public enum WeightMutation
    {
        None,
        Restricted,
        Unrestricted
    }

    public Accuracies UsedAccuracy { get; set; } = Accuracies.Micro;
    public Weighting UsedWeighting { get; set; } = Weighting.Computed;

    public bool UseFullInitialization { get; set; } = false;

    public WeightMutation WeightMutationToUse { get; set; } =
        WeightMutation.None;

    public IMetrics? Validate(IDataView validationData,
        IIndividualList individuals,
        string labelColumnName = DefaultColumnNames.Label)
    {
        var metrics = new Metrics();
        var labelColumn = validationData
            .GetColumnAsString(labelColumnName)
            .ToList();
        var labelDistribution = new float[data.Labels.Count];
        foreach (var label in labelColumn)
            labelDistribution[data.Labels.IndexOf(label)]++;
        AdaptLiterals(validationData);
        foreach (var individual in individuals)
        {
            ((LogicGpGenotype)individual.Genotype)
                .UpdatePredictionsRecursively();
            individual.Generation = 0;
            var fitnessValue = fitnessFunction.Evaluate(
                individual,
                validationData);

            var accuracy = 0.0;
            for (var i = 0; i < fitnessValue.Length - 1; i++)
                if (UsedAccuracy == Accuracies.Micro)
                    accuracy += fitnessValue[i];
                else
                    accuracy += labelDistribution[i] == 0
                        ? fitnessValue[i] == 0 ? 1.0 : 0.0
                        : fitnessValue[i] / labelDistribution[i];

            accuracy /= fitnessValue.Length - 1;

            //var accuracy = 0.0;
            //for (var i = 0; i < fitnessValue.Length - 1; i++)
            //    accuracy += fitnessValue[i];
            individual.LatestKnownFitness =
                [accuracy, fitnessValue[^1]];
        }

        return metrics;
    }

    public IMetrics? Test(IDataView testData)
    {
        return new Metrics();
    }

    public IIndividualList Train(IDataView trainData,
        bool firstTraining = true,
        string labelColumnName = DefaultColumnNames.Label,
        int generations = 10000
    )
    {
        if (firstTraining)
            PrepareForFirstTraining(trainData, labelColumnName);
        else
            PrepareForRetraining(trainData, labelColumnName);

        randomInitialization.Size = 2;
        generationStoppingCriterion.Limit = generations;
        selection.Size = 6;
        gp.SelectionForOperator = selection;
        gp.SelectionForSurvival = paretoFrontSelection;
        gp.PopulationManager = populationManager;
        gp.TrainingData = trainData;
        gp.Initialization = UseFullInitialization
            ? completeInitialization
            : randomInitialization;
        gp.Crossovers = [new LogicGpCrossover()];
        gp.Mutations =
        [
            new DeleteLiteral(), new InsertLiteral(),
            new InsertMonomial(), new ReplaceLiteral(), new DeleteMonomial()
        ];

        IMutation? weightMutation = WeightMutationToUse switch
        {
            WeightMutation.None => null,
            WeightMutation.Restricted => new ChangeWeightsRestricted(),
            WeightMutation.Unrestricted => new ChangeWeightsUnrestricted(),
            _ => null
        };

        if (weightMutation != null)
            ((IList)gp.Mutations).Add(weightMutation);
        fitnessFunction.LabelColumnName = data.Label;
        ((LogicGpPareto)fitnessFunction).Labels = data.Labels;
        gp.FitnessFunction = fitnessFunction;
        searchSpace.OutputColumn =
            IDataViewExtensions.GetColumnAsString(trainData, labelColumnName)
                .ToList();
        searchSpace.UsedWeighting = UsedWeighting;
        gp.SearchSpace = searchSpace;
        gp.StoppingCriteria = [generationStoppingCriterion];
        return gp.Run();
    }

    private void PrepareForFirstTraining(IDataView trainData,
        string labelColumnName)
    {
        data.Initialize(trainData, labelColumnName);
    }


    private void PrepareForRetraining(IDataView trainData,
        string labelColumnName)
    {
        AdaptLiterals(trainData);
    }

    private void AdaptLiterals(IDataView newData)
    {
        foreach (var literal in data.Literals)
            literal.GeneratePredictions(
                IDataViewExtensions.GetColumnAsString(newData, literal.Label)
                    .ToList());
    }
}