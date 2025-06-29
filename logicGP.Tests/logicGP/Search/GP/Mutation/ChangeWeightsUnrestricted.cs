using Italbytz.Adapters.Algorithms.AI.Search.GP.SearchSpace;
using Italbytz.AI.Search.GP.Individuals;
using Italbytz.AI.Search.GP.Mutation;

namespace Italbytz.Adapters.Algorithms.AI.Search.GP.Mutation;

public class ChangeWeightsUnrestricted : IMutation
{
    public IIndividualList Process(IIndividualList individuals)
    {
        var newPopulation = new Population();
        foreach (var individual in individuals)
        {
            var mutant = (IIndividual)individual.Clone();
            ((LogicGpGenotype)mutant.Genotype).RandomizeAMonomialWeight(false);
            newPopulation.Add(mutant);
        }

        return newPopulation;
    }
}