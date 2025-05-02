using Italbytz.Adapters.Algorithms.AI.Search.GP.Control;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Individuals;
using Italbytz.Ports.Algorithms.AI.Search.GP;
using Italbytz.Ports.Algorithms.AI.Search.GP.Individuals;
using Italbytz.Ports.Algorithms.AI.Search.GP.SearchSpace;

namespace Italbytz.Adapters.Algorithms.AI.Search.GP.SearchSpace;

/// <summary>
///     A class representing the search space for the LogicGP algorithm.
///     It implements the ISearchSpace interface and provides methods for
///     generating
///     random genotypes and starting populations.
/// </summary>
/// <remarks>
///     The LogicGpSearchSpace class is used to represent the search space for the
///     LogicGP algorithm.
///     It contains methods for generating random genotypes and starting
///     populations.
///     The class also provides properties for setting the output column and
///     weighting used in the algorithm.
/// </remarks>
/// <seealso cref="ISearchSpace" />
public class LogicGpSearchSpace(IGeneticProgram gp, DataManager data)
    : ISearchSpace
{
    public List<string>? OutputColumn { get; set; }
    public LogicGpAlgorithm.Weighting? UsedWeighting { get; set; }

    public IGenotype GetRandomGenotype()
    {
        var classes = data.Labels.Distinct().Count();
        return new LogicGpGenotype(classes, data, OutputColumn, data.Labels,
            UsedWeighting);
    }

    public IIndividualList GetAStartingPopulation()
    {
        var result = new Population();
        var classes = data.Labels.Distinct().Count();

        foreach (var polynomial in data.Literals
                     .Select(
                         literal =>
                             new LogicGpMonomial<string>([literal], classes,
                                 OutputColumn, data.Labels, UsedWeighting))
                     .Select(monomial =>
                         new LogicGpPolynomial<string>([monomial], classes,
                             OutputColumn, data.Labels, UsedWeighting)))
        {
            var newIndividual = new Individual(
                new LogicGpGenotype(polynomial, data, OutputColumn,
                    data.Labels, UsedWeighting),
                null);
            ((LogicGpGenotype)newIndividual.Genotype)
                .UpdatePredictionsRecursively();
            result.Add(newIndividual);
        }

        return result;
    }
}