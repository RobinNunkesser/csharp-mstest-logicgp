using Italbytz.Adapters.Algorithms.AI.Search.GP.SearchSpace;
using Italbytz.ML;
using Italbytz.Ports.Algorithms.AI.Search.GP.Fitness;
using Italbytz.Ports.Algorithms.AI.Search.GP.Individuals;
using Microsoft.ML;

namespace Italbytz.Adapters.Algorithms.AI.Search.GP.Fitness;

public class LogicGpPareto : IStaticMultiObjectiveFitnessFunction
{
    public List<string> Labels { get; set; }
    public int NumberOfObjectives { get; set; }
    public string LabelColumnName { get; set; } = DefaultColumnNames.Label;

    public double[] Evaluate(IIndividual individual, IDataView data)
    {
        NumberOfObjectives = individual.Genotype.Predictions[0].Length + 1;
        var predictions =
            ((LogicGpGenotype)individual.Genotype).PredictedClasses;
        var labels =
            data.GetColumnAsString(LabelColumnName)
                .ToList();
        var objectives = new double[NumberOfObjectives];
        for (var i = 0; i < predictions.Length; i++)
        {
            if (!predictions[i].Equals(labels[i])) continue;
            var index = Labels.IndexOf(labels[i]);
            objectives[index]++;
        }

        objectives[^1] = -individual.Size;

        individual.LatestKnownFitness = objectives;
        return objectives;
    }
}