using Italbytz.Adapters.Algorithms.AI.Search.GP.Individuals;
using Italbytz.Adapters.Algorithms.AI.Search.GP.SearchSpace;
using Italbytz.Ports.Algorithms.AI.Search.GP.Individuals;
using Italbytz.Ports.Algorithms.AI.Search.GP.Mutation;

namespace Italbytz.Adapters.Algorithms.AI.Search.GP.Mutation;

public class InsertLiteral : IMutation
{
    public IIndividualList Process(IIndividualList individuals)
    {
        var newPopulation = new Population();
        foreach (var individual in individuals)
        {
            var mutant = (IIndividual)individual.Clone();
            ((LogicGpGenotype)mutant.Genotype).InsertRandomLiteral();
            newPopulation.Add(mutant);
        }

        return newPopulation;
    }
}