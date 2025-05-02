using Italbytz.Adapters.Algorithms.AI.Search.GP.Individuals;
using Italbytz.Adapters.Algorithms.AI.Search.GP.SearchSpace;
using Italbytz.Ports.Algorithms.AI.Search.GP.Individuals;
using Italbytz.Ports.Algorithms.AI.Search.GP.Mutation;

namespace Italbytz.Adapters.Algorithms.AI.Search.GP.Mutation;

/// <summary>
///     A class representing a mutation operation that replaces a random literal in
///     a genotype.
///     It implements the IMutation interface and provides methods for processing a
///     list of individuals and replacing a random literal in each individual's
///     genotype.
/// </summary>
/// <remarks>
///     The ReplaceLiteral class is used to perform a mutation operation in the
///     LogicGP algorithm.
///     It replaces a random literal in the genotype of each individual in the
///     population.
///     The class implements the IMutation interface and provides a method for
///     processing a list of individuals.
///     The mutation operation is performed by calling the ReplaceRandomLiteral
///     method on the genotype of each individual.
/// </remarks>
/// <seealso cref="IMutation" />
public class ReplaceLiteral : IMutation
{
    public IIndividualList Process(IIndividualList individuals)
    {
        var newPopulation = new Population();
        foreach (var individual in individuals)
        {
            var mutant = (IIndividual)individual.Clone();
            ((LogicGpGenotype)mutant.Genotype).ReplaceRandomLiteral();
            newPopulation.Add(mutant);
        }

        return newPopulation;
    }
}